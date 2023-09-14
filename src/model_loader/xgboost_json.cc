/*!
 * Copyright (c) 2020-2023 by Contributors
 * \file xgboost_json.cc
 * \brief Model loader for XGBoost model (JSON)
 * \author Hyunsu Cho
 * \author William Hicks
 */
#include "detail/xgboost_json.h"

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/filereadstream.h>
#include <treelite/logging.h>
#include <treelite/model_loader.h>
#include <treelite/tree.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <variant>

#include "detail/common.h"
#include "detail/xgboost.h"

namespace {

template <typename StreamType, typename ErrorHandlerFunc>
std::unique_ptr<treelite::Model> ParseStream(std::unique_ptr<StreamType> input_stream,
    ErrorHandlerFunc error_handler, rapidjson::Document const& config);
template <class Iter, class T>
Iter BinarySearch(Iter begin, Iter end, T const& val);

}  // anonymous namespace

namespace treelite::model_loader {

std::unique_ptr<treelite::Model> LoadXGBoostModel(char const* filename, char const* config_json) {
  char read_buffer[65536];

#ifdef _WIN32
  FILE* fp = std::fopen(filename, "rb");
#else
  FILE* fp = std::fopen(filename, "r");
#endif
  if (!fp) {
    TREELITE_LOG(FATAL) << "Failed to open file '" << filename << "': " << std::strerror(errno);
  }

  auto input_stream
      = std::make_unique<rapidjson::FileReadStream>(fp, read_buffer, sizeof(read_buffer));
  auto error_handler = [fp](std::size_t offset) -> std::string {
    std::size_t cur = (offset >= 50 ? (offset - 50) : 0);
    std::fseek(fp, cur, SEEK_SET);
    int c;
    std::ostringstream oss, oss2;
    for (int i = 0; i < 100; ++i) {
      c = std::fgetc(fp);
      if (c == EOF) {
        break;
      }
      oss << static_cast<char>(c);
      if (cur == offset) {
        oss2 << "^";
      } else {
        oss2 << "~";
      }
      ++cur;
    }
    std::fclose(fp);
    return oss.str() + "\n" + oss2.str();
  };
  rapidjson::Document parsed_config;
  parsed_config.Parse(config_json);
  TREELITE_CHECK(!parsed_config.HasParseError())
      << "Error when parsing JSON config: offset " << parsed_config.GetErrorOffset() << ", "
      << rapidjson::GetParseError_En(parsed_config.GetParseError());
  auto parsed_model = ParseStream(std::move(input_stream), error_handler, parsed_config);
  std::fclose(fp);
  return parsed_model;
}

std::unique_ptr<treelite::Model> LoadXGBoostModelFromString(
    char const* json_str, std::size_t length, char const* config_json) {
  auto input_stream = std::make_unique<rapidjson::MemoryStream>(json_str, length);
  auto error_handler = [json_str](std::size_t offset) -> std::string {
    std::size_t cur = (offset >= 50 ? (offset - 50) : 0);
    std::ostringstream oss, oss2;
    for (int i = 0; i < 100; ++i) {
      if (!json_str[cur]) {
        break;
      }
      oss << json_str[cur];
      if (cur == offset) {
        oss2 << "^";
      } else {
        oss2 << "~";
      }
      ++cur;
    }
    return oss.str() + "\n" + oss2.str();
  };
  rapidjson::Document parsed_config;
  parsed_config.Parse(config_json);
  TREELITE_CHECK(!parsed_config.HasParseError())
      << "Error when parsing JSON config: offset " << parsed_config.GetErrorOffset() << ", "
      << rapidjson::GetParseError_En(parsed_config.GetParseError());
  return ParseStream(std::move(input_stream), error_handler, parsed_config);
}

namespace detail {

/******************************************************************************
 * BaseHandler
 * ***************************************************************************/

bool BaseHandler::pop_handler() {
  if (auto parent = delegator.lock()) {
    parent->pop_delegate();
    return true;
  } else {
    return false;
  }
}

bool BaseHandler::set_cur_key(char const* str, std::size_t length) {
  if (is_recognized_key(str)) {
    cur_key = std::string{str, length};
  } else if (allow_unknown_field_) {
    TREELITE_LOG(WARNING) << "Warning: Encountered unknown key \"" << str << "\"";
    cur_key = "";
    state_next_field_ignore_ = true;
  } else {
    // Extra field with unknown key is a fatal error if allow_unknown_field_ is not set
    TREELITE_LOG(ERROR) << "Error: key \"" << str << "\" is not recognized!";
    return false;
  }
  return true;
}

std::string const& BaseHandler::get_cur_key() {
  return cur_key;
}

bool BaseHandler::check_cur_key(std::string const& query_key) {
  return cur_key == query_key;
}

template <typename ValueType>
bool BaseHandler::assign_value(std::string const& key, ValueType&& value, ValueType& output) {
  if (check_cur_key(key)) {
    output = value;
    return true;
  } else {
    return false;
  }
}

template <typename ValueType>
bool BaseHandler::assign_value(std::string const& key, ValueType const& value, ValueType& output) {
  if (check_cur_key(key)) {
    output = value;
    return true;
  } else {
    return false;
  }
}

/******************************************************************************
 * IgnoreHandler
 * ***************************************************************************/
bool IgnoreHandler::Null() {
  return true;
}
bool IgnoreHandler::Bool(bool) {
  return true;
}
bool IgnoreHandler::Int(int) {
  return true;
}
bool IgnoreHandler::Uint(unsigned) {
  return true;
}
bool IgnoreHandler::Int64(int64_t) {
  return true;
}
bool IgnoreHandler::Uint64(uint64_t) {
  return true;
}
bool IgnoreHandler::Double(double) {
  return true;
}
bool IgnoreHandler::String(char const*, std::size_t, bool) {
  return true;
}
bool IgnoreHandler::StartObject() {
  return push_handler<IgnoreHandler>();
}
bool IgnoreHandler::Key(char const*, std::size_t, bool) {
  return true;
}
bool IgnoreHandler::StartArray() {
  return push_handler<IgnoreHandler>();
}

/******************************************************************************
 * TreeParamHandler
 * ***************************************************************************/
bool TreeParamHandler::String(char const* str, std::size_t, bool) {
  if (this->should_ignore_upcoming_value()) {
    return true;
  }
  // Key "num_deleted" deprecated but still present in some xgboost output
  return (check_cur_key("num_feature")
          || assign_value("num_nodes", std::stoi(str), output.num_nodes)
          || assign_value("size_leaf_vector", std::stoi(str), output.size_leaf_vector)
          || check_cur_key("num_deleted"));
}

bool TreeParamHandler::is_recognized_key(std::string const& key) {
  return (key == "num_feature" || key == "num_nodes" || key == "size_leaf_vector"
          || key == "num_deleted");
}

/******************************************************************************
 * RegTreeHandler
 * ***************************************************************************/
bool RegTreeHandler::StartArray() {
  if (this->should_ignore_upcoming_value()) {
    return push_handler<IgnoreHandler>();
  }
  return (push_key_handler<ArrayHandler<float>>("loss_changes", loss_changes)
          || push_key_handler<ArrayHandler<float>>("sum_hessian", sum_hessian)
          || push_key_handler<ArrayHandler<float>>("base_weights", base_weights)
          || push_key_handler<ArrayHandler<int>>("categories_segments", categories_segments)
          || push_key_handler<ArrayHandler<int>>("categories_sizes", categories_sizes)
          || push_key_handler<ArrayHandler<int>>("categories_nodes", categories_nodes)
          || push_key_handler<ArrayHandler<int>>("categories", categories)
          || push_key_handler<IgnoreHandler>("leaf_child_counts")
          || push_key_handler<ArrayHandler<int>>("left_children", left_children)
          || push_key_handler<ArrayHandler<int>>("right_children", right_children)
          || push_key_handler<ArrayHandler<int>>("parents", parents)
          || push_key_handler<ArrayHandler<int>>("split_indices", split_indices)
          || push_key_handler<ArrayHandler<int>>("split_type", split_type)
          || push_key_handler<ArrayHandler<float>>("split_conditions", split_conditions)
          || push_key_handler<ArrayHandler<bool>>("default_left", default_left));
}

bool RegTreeHandler::StartObject() {
  if (this->should_ignore_upcoming_value()) {
    return push_handler<IgnoreHandler>();
  }
  return push_key_handler<TreeParamHandler, ParsedRegTreeParams>("tree_param", reg_tree_params);
}

bool RegTreeHandler::Uint(unsigned) {
  if (this->should_ignore_upcoming_value()) {
    return true;
  }
  return check_cur_key("id");
}

bool RegTreeHandler::EndObject(std::size_t) {
  output.Init(true);
  auto const num_nodes = reg_tree_params.num_nodes;
  if (split_type.empty()) {
    split_type.resize(num_nodes, xgboost::FeatureType::kNumerical);
  }
  if (static_cast<std::size_t>(num_nodes) != loss_changes.size()) {
    TREELITE_LOG(ERROR) << "Field loss_changes has an incorrect dimension. Expected: " << num_nodes
                        << ", Actual: " << loss_changes.size();
    return false;
  }
  if (static_cast<std::size_t>(num_nodes) != sum_hessian.size()) {
    TREELITE_LOG(ERROR) << "Field sum_hessian has an incorrect dimension. Expected: " << num_nodes
                        << ", Actual: " << sum_hessian.size();
    return false;
  }
  if (static_cast<std::size_t>(num_nodes) != base_weights.size()) {
    TREELITE_LOG(ERROR) << "Field base_weights has an incorrect dimension. Expected: " << num_nodes
                        << ", Actual: " << base_weights.size();
    return false;
  }
  if (static_cast<std::size_t>(num_nodes) != left_children.size()) {
    TREELITE_LOG(ERROR) << "Field left_children has an incorrect dimension. Expected: " << num_nodes
                        << ", Actual: " << left_children.size();
    return false;
  }
  if (static_cast<std::size_t>(num_nodes) != right_children.size()) {
    TREELITE_LOG(ERROR) << "Field right_children has an incorrect dimension. Expected: "
                        << num_nodes << ", Actual: " << right_children.size();
    return false;
  }
  if (static_cast<std::size_t>(num_nodes) != parents.size()) {
    TREELITE_LOG(ERROR) << "Field parents has an incorrect dimension. Expected: " << num_nodes
                        << ", Actual: " << parents.size();
    return false;
  }
  if (static_cast<std::size_t>(num_nodes) != split_indices.size()) {
    TREELITE_LOG(ERROR) << "Field split_indices has an incorrect dimension. Expected: " << num_nodes
                        << ", Actual: " << split_indices.size();
    return false;
  }
  if (static_cast<std::size_t>(num_nodes) != split_type.size()) {
    TREELITE_LOG(ERROR) << "Field split_type has an incorrect dimension. Expected: " << num_nodes
                        << ", Actual: " << split_type.size();
    return false;
  }
  if (static_cast<std::size_t>(num_nodes) != split_conditions.size()) {
    TREELITE_LOG(ERROR) << "Field split_conditions has an incorrect dimension. Expected: "
                        << num_nodes << ", Actual: " << split_conditions.size();
    return false;
  }
  if (static_cast<std::size_t>(num_nodes) != default_left.size()) {
    TREELITE_LOG(ERROR) << "Field default_left has an incorrect dimension. Expected: " << num_nodes
                        << ", Actual: " << default_left.size();
    return false;
  }

  std::queue<std::pair<int, int>> Q;  // (old ID, new ID) pair
  if (num_nodes > 0) {
    Q.emplace(0, 0);
  }
  while (!Q.empty()) {
    int old_id, new_id;
    std::tie(old_id, new_id) = Q.front();
    Q.pop();

    if (left_children[old_id] == -1) {
      auto const size_leaf_vector = reg_tree_params.size_leaf_vector;
      if (size_leaf_vector > 1) {
        // Vector output
        std::vector<float> leafvec(size_leaf_vector);
        std::transform(&base_weights[old_id * size_leaf_vector],
            &base_weights[(old_id + 1) * size_leaf_vector], leafvec.begin(),
            [](float e) { return static_cast<float>(e); });
        output.SetLeafVector(new_id, leafvec);
      } else {
        // Scalar leaf output
        output.SetLeaf(new_id, static_cast<float>(split_conditions[old_id]));
      }
    } else {
      output.AddChilds(new_id);
      if (split_type[old_id] == xgboost::FeatureType::kCategorical) {
        auto categorical_split_loc
            = BinarySearch(categories_nodes.begin(), categories_nodes.end(), old_id);
        TREELITE_CHECK(categorical_split_loc != categories_nodes.end())
            << "Could not find record for the categorical split in node " << old_id;
        auto categorical_split_id = std::distance(categories_nodes.begin(), categorical_split_loc);
        int offset = categories_segments[categorical_split_id];
        int num_categories = categories_sizes[categorical_split_id];
        std::vector<std::uint32_t> right_categories;
        right_categories.reserve(num_categories);
        for (int i = 0; i < num_categories; ++i) {
          right_categories.push_back(static_cast<std::uint32_t>(categories[offset + i]));
        }
        output.SetCategoricalTest(
            new_id, split_indices[old_id], default_left[old_id], right_categories, true);
      } else {
        output.SetNumericalTest(new_id, split_indices[old_id],
            static_cast<float>(split_conditions[old_id]), default_left[old_id],
            treelite::Operator::kLT);
      }
      output.SetGain(new_id, loss_changes[old_id]);
      Q.emplace(left_children[old_id], output.LeftChild(new_id));
      Q.emplace(right_children[old_id], output.RightChild(new_id));
    }
    output.SetSumHess(new_id, sum_hessian[old_id]);
  }
  return pop_handler();
}

bool RegTreeHandler::is_recognized_key(std::string const& key) {
  return (key == "loss_changes" || key == "sum_hessian" || key == "base_weights"
          || key == "categories_segments" || key == "categories_sizes" || key == "categories_nodes"
          || key == "categories" || key == "leaf_child_counts" || key == "left_children"
          || key == "right_children" || key == "parents" || key == "split_indices"
          || key == "split_type" || key == "split_conditions" || key == "default_left"
          || key == "tree_param" || key == "id");
}

/******************************************************************************
 * GBTreeHandler
 * ***************************************************************************/
bool GBTreeModelHandler::StartArray() {
  if (this->should_ignore_upcoming_value()) {
    return push_handler<IgnoreHandler>();
  }
  auto& trees = std::get<ModelPreset<float, float>>(output.model->variant_).trees;
  return (push_key_handler<ArrayHandler<treelite::Tree<float, float>, RegTreeHandler>,
              std::vector<treelite::Tree<float, float>>>("trees", trees)
          || push_key_handler<ArrayHandler<int>, std::vector<int>>("tree_info", output.tree_info)
          || push_key_handler<IgnoreHandler>("iteration_indptr"));
}

bool GBTreeModelHandler::StartObject() {
  if (this->should_ignore_upcoming_value()) {
    return push_handler<IgnoreHandler>();
  }
  return push_key_handler<IgnoreHandler>("gbtree_model_param");
}

bool GBTreeModelHandler::EndObject(std::size_t) {
  // We will assume a well-formed XGBoost model, where all leafs produce an output of identical
  // dimension. It will slow down parsing if we were to actually check all nodes.
  auto& trees = std::get<ModelPreset<float, float>>(output.model->variant_).trees;
  output.size_leaf_vector = 1;
  for (Tree<float, float> const& tree : trees) {
    for (int node_id = 0; node_id < tree.num_nodes; ++node_id) {
      if (tree.IsLeaf(node_id)) {
        if (tree.HasLeafVector(node_id)) {
          output.size_leaf_vector = static_cast<int>(tree.LeafVector(node_id).size());
        }
        break;
      }
    }
  }
  return pop_handler();
}

bool GBTreeModelHandler::is_recognized_key(std::string const& key) {
  return (key == "trees" || key == "tree_info" || key == "gbtree_model_param"
          || key == "iteration_indptr");
}

/******************************************************************************
 * GradientBoosterHandler
 * ***************************************************************************/
bool GradientBoosterHandler::String(char const* str, std::size_t length, bool) {
  if (this->should_ignore_upcoming_value()) {
    return true;
  }
  if (assign_value("name", std::string{str, length}, name)) {
    if (name == "gbtree" || name == "dart") {
      return true;
    } else {
      TREELITE_LOG(ERROR) << "Only GBTree or DART boosters are currently supported.";
      return false;
    }
  } else {
    return false;
  }
}

bool GradientBoosterHandler::StartObject() {
  if (this->should_ignore_upcoming_value()) {
    return push_handler<IgnoreHandler>();
  }
  if (push_key_handler<GBTreeModelHandler, ParsedXGBoostModel>("model", output)) {
    return true;
  } else if (push_key_handler<GradientBoosterHandler, ParsedXGBoostModel>("gbtree", output)) {
    // "dart" booster contains a standard gbtree under ["gradient_booster"]["gbtree"]["model"].
    return true;
  } else {
    TREELITE_LOG(ERROR) << "Key \"" << get_cur_key()
                        << "\" not recognized. Is this a GBTree-type booster?";
    return false;
  }
}

bool GradientBoosterHandler::StartArray() {
  if (this->should_ignore_upcoming_value()) {
    return push_handler<IgnoreHandler>();
  }
  return push_key_handler<ArrayHandler<float>, std::vector<float>>("weight_drop", weight_drop);
}

bool GradientBoosterHandler::EndObject([[maybe_unused]] std::size_t memberCount) {
  if (name == "dart" && !weight_drop.empty()) {
    TREELITE_CHECK_EQ(output.size_leaf_vector, 1)
        << "Dart with vector-leaf output is not yet supported";
    // Fold weight drop into leaf value for dart models.
    auto& trees = std::get<ModelPreset<float, float>>(output.model->variant_).trees;
    TREELITE_CHECK_EQ(trees.size(), weight_drop.size());
    for (std::size_t i = 0; i < trees.size(); ++i) {
      for (int nid = 0; nid < trees[i].num_nodes; ++nid) {
        if (trees[i].IsLeaf(nid)) {
          trees[i].SetLeaf(nid, static_cast<float>(weight_drop[i] * trees[i].LeafValue(nid)));
        }
      }
    }
  }
  return pop_handler();
}

bool GradientBoosterHandler::is_recognized_key(std::string const& key) {
  return (key == "name" || key == "model" || key == "gbtree" || key == "weight_drop");
}

/******************************************************************************
 * ObjectiveHandler
 * ***************************************************************************/
bool ObjectiveHandler::StartObject() {
  if (this->should_ignore_upcoming_value()) {
    return push_handler<IgnoreHandler>();
  }
  // pseduo_huber_param is a mispelled variant of pseudo_huber_param; it's used in a past version
  // of XGBoost
  return (push_key_handler<IgnoreHandler>("reg_loss_param")
          || push_key_handler<IgnoreHandler>("poisson_regression_param")
          || push_key_handler<IgnoreHandler>("tweedie_regression_param")
          || push_key_handler<IgnoreHandler>("softmax_multiclass_param")
          || push_key_handler<IgnoreHandler>("lambda_rank_param")
          || push_key_handler<IgnoreHandler>("aft_loss_param")
          || push_key_handler<IgnoreHandler>("pseduo_huber_param")
          || push_key_handler<IgnoreHandler>("pseudo_huber_param"));
}

bool ObjectiveHandler::String(char const* str, std::size_t length, bool) {
  if (this->should_ignore_upcoming_value()) {
    return true;
  }
  return assign_value("name", std::string{str, length}, output);
}

bool ObjectiveHandler::is_recognized_key(std::string const& key) {
  return (key == "reg_loss_param" || key == "poisson_regression_param"
          || key == "tweedie_regression_param" || key == "softmax_multiclass_param"
          || key == "lambda_rank_param" || key == "aft_loss_param" || key == "pseduo_huber_param"
          || key == "pseudo_huber_param" || key == "name");
}

/******************************************************************************
 * LearnerParamHandler
 * ***************************************************************************/
bool LearnerParamHandler::String(char const* str, std::size_t, bool) {
  if (this->should_ignore_upcoming_value()) {
    return true;
  }
  // For now, XGBoost always outputs a scalar base_score
  return (
      assign_value("base_score", static_cast<float>(std::strtod(str, nullptr)), output.base_score)
      || assign_value("num_class", std::max(std::stoi(str), 1), output.num_class)
      || assign_value("num_target", static_cast<std::uint32_t>(std::stoi(str)), output.num_target)
      || assign_value("num_feature", std::stoi(str), output.num_feature)
      || assign_value(
          "boost_from_average", static_cast<bool>(std::stoi(str)), output.boost_from_average));
}

bool LearnerParamHandler::is_recognized_key(std::string const& key) {
  return (key == "num_target" || key == "base_score" || key == "num_class" || key == "num_feature"
          || key == "boost_from_average");
}

/******************************************************************************
 * LearnerHandler
 * ***************************************************************************/
bool LearnerHandler::StartObject() {
  if (this->should_ignore_upcoming_value()) {
    return push_handler<IgnoreHandler>();
  }
  // "attributes" key is not documented in schema
  return (
      push_key_handler<LearnerParamHandler, ParsedLearnerParams>(
          "learner_model_param", learner_params)
      || push_key_handler<GradientBoosterHandler, ParsedXGBoostModel>("gradient_booster", output)
      || push_key_handler<ObjectiveHandler, std::string>("objective", objective)
      || push_key_handler<IgnoreHandler>("attributes"));
}

bool LearnerHandler::EndObject(std::size_t) {
  auto const num_tree = output.num_tree;

  output.model->num_feature = learner_params.num_feature;
  output.model->average_tree_output = false;

  // For now, XGBoost always outputs a scalar base_score
  output.model->base_scores.Resize(1);
  output.model->base_scores[0] = learner_params.base_score;

  output.model->num_target = learner_params.num_target;

  // Save the objective name
  std::string const pred_transform = xgboost::GetPredTransform(objective);
  output.model->pred_transform = pred_transform;
  output.objective_name = objective;

  if (learner_params.num_class > 1) {  // Multi-class classifier
    // For now, XGBoost does not support multi-target models for multi-class classification
    // So if num_class > 1, we can assume num_target == 1
    TREELITE_CHECK_EQ(learner_params.num_target, 1)
        << "XGBoost does not support multi-target models for multi-class classification";
    output.model->num_class
        = std::vector<std::uint32_t>{static_cast<std::uint32_t>(learner_params.num_class)};
    output.model->task_type = TaskType::kMultiClf;
    output.model->target_id.Resize(num_tree);
    std::fill_n(output.model->target_id.Data(), num_tree, 0);

    output.model->class_id.Resize(num_tree);
    if (output.size_leaf_vector > 1) {
      // Vector-leaf output
      std::fill_n(output.model->class_id.Data(), num_tree, -1);
    } else {
      // Grove per class: i-th tree produces output for class (i % num_class)
      // Note: num_parallel_tree can change this behavior, so it's best to go with
      // tree_info field provided by XGBoost
      for (std::int32_t tree_id = 0; tree_id < num_tree; ++tree_id) {
        output.model->class_id[tree_id] = static_cast<std::int32_t>(output.tree_info[tree_id]);
      }
    }
    output.model->leaf_vector_shape
        = std::vector<std::uint32_t>{1, static_cast<std::uint32_t>(output.size_leaf_vector)};
  } else {
    // Binary classifier or regressor
    if (StringStartsWith(output.objective_name, "binary:")) {
      output.model->task_type = TaskType::kBinaryClf;
    } else if (StringStartsWith(output.objective_name, "rank:")) {
      output.model->task_type = TaskType::kLearningToRank;
    } else {
      output.model->task_type = TaskType::kRegressor;
    }
    const std::uint32_t num_target = learner_params.num_target;
    output.model->num_class.Resize(num_target);
    std::fill_n(output.model->num_class.Data(), num_target, 1);
    output.model->class_id.Resize(num_tree);
    std::fill_n(output.model->class_id.Data(), num_tree, 0);
    output.model->target_id.Resize(num_tree);
    if (output.size_leaf_vector > 1) {
      // Vector-leaf output
      std::fill_n(output.model->target_id.Data(), num_tree, -1);
      TREELITE_CHECK_EQ(num_target, static_cast<std::uint32_t>(output.size_leaf_vector));
      output.model->leaf_vector_shape = std::vector<std::uint32_t>{num_target, 1};
    } else {
      // Grove per target: i-th tree produces output for target (i % num_target)
      for (std::int32_t tree_id = 0; tree_id < num_tree; ++tree_id) {
        // Validate tree_info
        auto const grove_id = static_cast<std::uint32_t>(output.tree_info[tree_id]);
        auto const expected_grove_id = static_cast<std::uint32_t>(tree_id) % num_target;
        TREELITE_CHECK_EQ(grove_id, expected_grove_id)
            << "tree_info for Tree " << tree_id << " is not valid! "
            << "Expected: " << expected_grove_id << ", Got: " << grove_id;
        output.model->target_id[tree_id] = static_cast<std::int32_t>(grove_id);
      }
      output.model->leaf_vector_shape = std::vector<std::uint32_t>{1, 1};
    }
  }
  // Before XGBoost 1.0.0, the base score saved in model is a transformed value.  After
  // 1.0 it's the original value provided by user.
  bool const need_transform_to_margin = output.version.empty() || output.version[0] >= 1;
  if (need_transform_to_margin) {
    learner_params.base_score
        = xgboost::TransformBaseScoreToMargin(pred_transform, learner_params.base_score);
  }
  // For now, XGBoost produces a scalar base_score
  // Assume: Either num_target or num_class must be 1
  TREELITE_CHECK(learner_params.num_target == 1 || learner_params.num_class == 1);
  std::size_t const len_base_scores = learner_params.num_target * learner_params.num_class;
  output.model->base_scores.Resize(len_base_scores);
  std::fill_n(output.model->base_scores.Data(), len_base_scores, learner_params.base_score);

  return pop_handler();
}

bool LearnerHandler::StartArray() {
  if (this->should_ignore_upcoming_value()) {
    return push_handler<IgnoreHandler>();
  }
  return (push_key_handler<IgnoreHandler>("feature_names")
          || push_key_handler<IgnoreHandler>("feature_types"));
}

bool LearnerHandler::is_recognized_key(std::string const& key) {
  return (key == "learner_model_param" || key == "gradient_booster" || key == "objective"
          || key == "attributes" || key == "feature_names" || key == "feature_types");
}

/******************************************************************************
 * XGBoostCheckpointHandler
 * ***************************************************************************/

bool XGBoostCheckpointHandler::StartArray() {
  if (this->should_ignore_upcoming_value()) {
    return push_handler<IgnoreHandler>();
  }
  return push_key_handler<ArrayHandler<unsigned>, std::vector<unsigned>>("version", output.version);
}

bool XGBoostCheckpointHandler::StartObject() {
  if (this->should_ignore_upcoming_value()) {
    return push_handler<IgnoreHandler>();
  }
  return push_key_handler<LearnerHandler, ParsedXGBoostModel>("learner", output);
}

bool XGBoostCheckpointHandler::is_recognized_key(std::string const& key) {
  return (key == "version" || key == "learner");
}

/******************************************************************************
 * XGBoostModelHandler
 * ***************************************************************************/
bool XGBoostModelHandler::StartArray() {
  if (this->should_ignore_upcoming_value()) {
    return push_handler<IgnoreHandler>();
  }
  return push_key_handler<ArrayHandler<unsigned>, std::vector<unsigned>>("version", output.version);
}

bool XGBoostModelHandler::StartObject() {
  if (this->should_ignore_upcoming_value()) {
    return push_handler<IgnoreHandler>();
  }
  return (push_key_handler<LearnerHandler, ParsedXGBoostModel>("learner", output)
          || push_key_handler<IgnoreHandler>("Config")
          || push_key_handler<XGBoostCheckpointHandler, ParsedXGBoostModel>("Model", output));
}

bool XGBoostModelHandler::is_recognized_key(std::string const& key) {
  return (key == "version" || key == "learner" || key == "Config" || key == "Model");
}

/******************************************************************************
 * RootHandler
 * ***************************************************************************/
bool RootHandler::StartObject() {
  if (this->should_ignore_upcoming_value()) {
    return push_handler<IgnoreHandler>();
  }
  return push_handler<XGBoostModelHandler, ParsedXGBoostModel>(output);
}

/******************************************************************************
 * DelegatedHandler
 * ***************************************************************************/
ParsedXGBoostModel DelegatedHandler::get_result() {
  return std::move(result);
}
bool DelegatedHandler::Null() {
  return delegates.top()->Null();
}
bool DelegatedHandler::Bool(bool b) {
  return delegates.top()->Bool(b);
}
bool DelegatedHandler::Int(int i) {
  return delegates.top()->Int(i);
}
bool DelegatedHandler::Uint(unsigned u) {
  return delegates.top()->Uint(u);
}
bool DelegatedHandler::Int64(int64_t i) {
  return delegates.top()->Int64(i);
}
bool DelegatedHandler::Uint64(uint64_t u) {
  return delegates.top()->Uint64(u);
}
bool DelegatedHandler::Double(double d) {
  return delegates.top()->Double(d);
}
bool DelegatedHandler::String(char const* str, std::size_t length, bool copy) {
  return delegates.top()->String(str, length, copy);
}
bool DelegatedHandler::StartObject() {
  return delegates.top()->StartObject();
}
bool DelegatedHandler::Key(char const* str, std::size_t length, bool copy) {
  return delegates.top()->Key(str, length, copy);
}
bool DelegatedHandler::EndObject(std::size_t memberCount) {
  return delegates.top()->EndObject(memberCount);
}
bool DelegatedHandler::StartArray() {
  return delegates.top()->StartArray();
}
bool DelegatedHandler::EndArray(std::size_t elementCount) {
  return delegates.top()->EndArray(elementCount);
}

}  // namespace detail
}  // namespace treelite::model_loader

namespace {

/*!
 * \brief Perform binary search on the range [begin, end).
 * \param begin Beginning of the search range
 * \param end End of the search range
 * \param val Value being searched
 * \return Iterator pointing to the value if found; end if value not found.
 * \tparam Iter Type of iterator
 * \tparam T Type of elements
 */
template <class Iter, class T>
Iter BinarySearch(Iter begin, Iter end, T const& val) {
  Iter i = std::lower_bound(begin, end, val);
  if (i != end && val == *i) {
    return i;  // found
  } else {
    return end;  // not found
  }
}

template <typename StreamType, typename ErrorHandlerFunc>
std::unique_ptr<treelite::Model> ParseStream(std::unique_ptr<StreamType> input_stream,
    ErrorHandlerFunc error_handler, rapidjson::Document const& config) {
  std::shared_ptr<treelite::model_loader::detail::DelegatedHandler> handler
      = treelite::model_loader::detail::DelegatedHandler::create(config);
  rapidjson::Reader reader;

  rapidjson::ParseResult result
      = reader.Parse<rapidjson::ParseFlag::kParseNanAndInfFlag>(*input_stream, *handler);
  if (!result) {
    auto const error_code = result.Code();
    const std::size_t offset = result.Offset();
    std::string diagnostic = error_handler(offset);
    TREELITE_LOG(FATAL) << "Provided JSON could not be parsed as XGBoost model. "
                        << "Parsing error at offset " << offset << ": "
                        << rapidjson::GetParseError_En(error_code) << "\n"
                        << diagnostic;
  }
  treelite::model_loader::detail::ParsedXGBoostModel parsed = handler->get_result();
  return std::move(parsed.model);
}

}  // anonymous namespace
