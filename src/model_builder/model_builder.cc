/*!
 * Copyright (c) 2023 by Contributors
 * \file model_builder.cc
 * \brief C++ API for constructing Model objects
 * \author Hyunsu Cho
 */
#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <treelite/enum/typeinfo.h>
#include <treelite/model_builder.h>
#include <treelite/tree.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace treelite::model_builder {

namespace detail {

void ConfigurePredTransform(Model* model, PredTransformFunc pred_transform) {
  rapidjson::Document config;
  config.Parse(pred_transform.config_json.c_str());
  TREELITE_CHECK(!config.HasParseError())
      << "Error when parsing JSON config: offset " << config.GetErrorOffset() << ", "
      << rapidjson::GetParseError_En(config.GetParseError());
  if (pred_transform.pred_transform_name == "sigmoid") {
    auto itr = config.FindMember("sigmoid_alpha");
    if (itr != config.MemberEnd() && itr->value.IsFloat()) {
      model->sigmoid_alpha = itr->value.GetFloat();
    } else {
      model->sigmoid_alpha = 1.0f;
    }
  }
  if (pred_transform.pred_transform_name == "exponential_standard_ratio") {
    auto itr = config.FindMember("ratio_c");
    if (itr != config.MemberEnd() && itr->value.IsFloat()) {
      model->ratio_c = itr->value.GetFloat();
    } else {
      model->ratio_c = 1.0f;
    }
  }
}

template <typename ThresholdT, typename LeafOutputT>
class ModelBuilderImpl : public ModelBuilder {
 public:
  ModelBuilderImpl(Metadata const& metadata, TreeAnnotation const& tree_annotation,
      PredTransformFunc const& pred_transform, std::vector<double> const& base_scores,
      std::optional<std::string> const& attributes)
      : threshold_type_(TypeInfoFromType<ThresholdT>()),
        leaf_output_type_(TypeInfoFromType<LeafOutputT>()),
        expected_num_tree_{tree_annotation.num_tree},
        model_{Model::Create<ThresholdT, LeafOutputT>()},
        current_tree_{} {
    const std::uint32_t num_tree = tree_annotation.num_tree;
    const std::uint32_t num_target = metadata.num_target;

    model_->num_feature = metadata.num_feature;
    model_->task_type = metadata.task_type;
    model_->average_tree_output = metadata.average_tree_output;
    model_->num_target = num_target;
    model_->num_class = metadata.num_class;
    model_->leaf_vector_shape = std::vector<std::uint32_t>(
        metadata.leaf_vector_shape.begin(), metadata.leaf_vector_shape.end());

    // Validate target_id and class_id
    for (std::uint32_t i = 0; i < num_tree; ++i) {
      TREELITE_CHECK_LT(tree_annotation.target_id[i], num_target);
    }
    model_->target_id = tree_annotation.target_id;
    for (std::uint32_t i = 0; i < num_tree; ++i) {
      TREELITE_CHECK_LT(
          tree_annotation.class_id[i], metadata.num_class[tree_annotation.target_id[i]]);
    }
    model_->class_id = tree_annotation.class_id;

    model_->pred_transform = pred_transform.pred_transform_name;
    detail::ConfigurePredTransform(model_.get(), pred_transform);

    const std::uint32_t max_num_class
        = *std::max_element(metadata.num_class.begin(), metadata.num_class.end());
    TREELITE_CHECK_EQ(base_scores.size(), num_target * max_num_class);
    model_->base_scores = base_scores;
    if (attributes) {
      model_->attributes = attributes.value();
    }
  }

  void StartTree() override {
    current_tree_ = Tree<ThresholdT, LeafOutputT>();
    current_tree_.Init();
  }

  void EndTree() override {
    // TODO(hcho3): Add some validation logic
    auto& trees = std::get<ModelPreset<ThresholdT, LeafOutputT>>(model_->variant_).trees;
    trees.push_back(std::move(current_tree_));
  }

  void StartNode(int node_key) override {
    int node_id = current_tree_.AllocNode();
    current_node_id_ = node_id;
    node_id_map_[node_key] = node_id;
  }
  void EndNode() override {}
  void NumericalTest(std::int32_t split_index, double threshold, bool default_left, Operator cmp,
      int left_child_key, int right_child_key) override {}
  void CategoricalTest(std::int32_t split_index, bool default_left,
      std::vector<std::uint32_t> const& category_list, bool category_list_right_child,
      int left_child_key, int right_child_key) override {}
  void LeafScalar(double leaf_value) override {}
  void LeafVector(std::vector<float> const& leaf_vector) override {}
  void LeafVector(std::vector<double> const& leaf_vector) override {}
  void Gain(double gain) override {}
  void DataCount(std::uint64_t data_count) override {}
  void SumHess(double sum_hess) override {}

  std::unique_ptr<Model> CommitModel() override {
    TREELITE_CHECK_EQ(model_->GetNumTree(), expected_num_tree_)
        << "Expected " << expected_num_tree_ << " trees but only got " << model_->GetNumTree()
        << " trees instead";
    return std::move(model_);
  }

 private:
  TypeInfo threshold_type_;
  TypeInfo leaf_output_type_;
  std::uint32_t expected_num_tree_;
  std::unique_ptr<Model> model_;
  Tree<ThresholdT, LeafOutputT> current_tree_;
  std::map<int, int> node_id_map_;  // user-defined ID -> internal ID
  int current_node_id_;
};

}  // namespace detail

std::unique_ptr<ModelBuilder> InitializeModel(TypeInfo threshold_type, TypeInfo leaf_output_type,
    Metadata const& metadata, TreeAnnotation const& tree_annotation,
    PredTransformFunc const& pred_transform, std::vector<double> const& base_scores,
    std::optional<std::string> const& attributes) {
  TREELITE_CHECK(threshold_type == TypeInfo::kFloat32 || threshold_type == TypeInfo::kFloat64);
  TREELITE_CHECK(leaf_output_type == threshold_type);
  if (threshold_type == TypeInfo::kFloat32) {
    return std::make_unique<detail::ModelBuilderImpl<float, float>>(
        metadata, tree_annotation, pred_transform, base_scores, attributes);
  } else {
    return std::make_unique<detail::ModelBuilderImpl<double, double>>(
        metadata, tree_annotation, pred_transform, base_scores, attributes);
  }
}

}  // namespace treelite::model_builder