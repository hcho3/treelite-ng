/*!
 * Copyright (c) 2021-2023 by Contributors
 * \file sklearn.cc
 * \brief Frontend for scikit-learn models
 * \author Hyunsu Cho
 */
#include <treelite/enum/operator.h>
#include <treelite/enum/task_type.h>
#include <treelite/enum/typeinfo.h>
#include <treelite/logging.h>
#include <treelite/model_builder.h>
#include <treelite/model_loader.h>
#include <treelite/tree.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>

namespace treelite::model_loader::sklearn {

namespace detail {

class RandomForestRegressorMixIn {
 public:
  void HandleMetadata(model_builder::ModelBuilder& builder, int n_trees, int n_features,
      int n_targets, [[maybe_unused]] std::int32_t const* n_classes) {
    n_targets_ = n_targets;
    model_builder::Metadata metadata{n_features, TaskType::kRegressor, true,
        static_cast<std::int32_t>(n_targets), std::vector<std::int32_t>(n_targets, 1),
        {n_targets, 1}};
    std::vector<std::int32_t> const target_id(n_trees, (n_targets > 1 ? -1 : 0));
    model_builder::TreeAnnotation tree_annotation{
        n_trees, target_id, std::vector<std::int32_t>(n_trees, 0)};
    model_builder::PredTransformFunc pred_transform{"identity"};
    std::vector<double> base_scores(n_targets, 0.0);
    builder.InitializeMetadata(
        metadata, tree_annotation, pred_transform, base_scores, std::nullopt);
  }

  void HandleLeafNode(model_builder::ModelBuilder& builder, int tree_id, int node_id,
      double const** value, [[maybe_unused]] std::int32_t const* n_classes) const {
    TREELITE_CHECK_GT(n_targets_, 0)
        << "n_targets not yet initialized. Was HandleMetadata() called?";
    if (n_targets_ == 1) {
      builder.LeafScalar(value[tree_id][node_id]);
    } else {
      std::vector<double> leafvec(
          &value[tree_id][node_id * n_targets_], &value[tree_id][(node_id + 1) * n_targets_]);
      builder.LeafVector(leafvec);
    }
  }

 private:
  int n_targets_{-1};
};

// Note: Here, we will treat binary classifiers as if they are multi-class classifiers with
// n_classes=2.
class RandomForestClassifierMixIn {
 public:
  void HandleMetadata(model_builder::ModelBuilder& builder, int n_trees, int n_features,
      int n_targets, std::int32_t const* n_classes) {
    n_targets_ = n_targets;
    std::vector<std::int32_t> n_classes_(n_classes, n_classes + n_targets);
    if (!std::all_of(n_classes_.begin(), n_classes_.end(), [](auto e) { return e >= 2; })) {
      TREELITE_LOG(FATAL)
          << "All elements in n_classes must be at least 2. "
          << "Note: For sklearn RandomForestClassifier, binary classifiers will have n_classes=2.";
    }
    max_num_class_ = *std::max_element(n_classes_.begin(), n_classes_.end());
    model_builder::Metadata metadata{n_features, TaskType::kMultiClf, true,
        static_cast<std::int32_t>(n_targets), n_classes_, {n_targets, max_num_class_}};
    model_builder::TreeAnnotation tree_annotation{
        n_trees, std::vector<std::int32_t>(n_trees, -1), std::vector<std::int32_t>(n_trees, -1)};
    model_builder::PredTransformFunc pred_transform{"identity_multiclass"};
    std::vector<double> base_scores(n_targets * max_num_class_, 0.0);
    builder.InitializeMetadata(
        metadata, tree_annotation, pred_transform, base_scores, std::nullopt);
  }

  void HandleLeafNode(model_builder::ModelBuilder& builder, int tree_id, int node_id,
      double const** value, [[maybe_unused]] int const* n_classes) const {
    TREELITE_CHECK_GT(n_targets_, 0)
        << "n_targets not yet initialized. Was HandleMetadata() called?";
    TREELITE_CHECK_GT(max_num_class_, 0)
        << "max_num_class not yet initialized. Was HandleMetadata() called?";
    std::vector<double> leafvec(&value[tree_id][node_id * n_targets_ * max_num_class_],
        &value[tree_id][(node_id + 1) * max_num_class_]);
    // Compute the probability distribution over label classes
    double const norm_factor = std::accumulate(leafvec.begin(), leafvec.end(), 0.0);
    std::for_each(leafvec.begin(), leafvec.end(), [norm_factor](double& e) { e /= norm_factor; });
    builder.LeafVector(leafvec);
  }

 private:
  int n_targets_{-1};
  std::int32_t max_num_class_{-1};
};

class IsolationForestMixIn {
 public:
  explicit IsolationForestMixIn(double ratio_c) : ratio_c_{ratio_c} {}

  void HandleMetadata(model_builder::ModelBuilder& builder, int n_trees, int n_features,
      [[maybe_unused]] int n_targets, [[maybe_unused]] std::int32_t const* n_classes) {
    model_builder::Metadata metadata{n_features, TaskType::kIsolationForest, true, 1, {1}, {1, 1}};
    model_builder::TreeAnnotation tree_annotation{
        n_trees, std::vector<std::int32_t>(n_trees, 0), std::vector<std::int32_t>(n_trees, 0)};

    std::ostringstream oss;
    oss << "{\"ratio_c\": " << ratio_c_ << "}";
    auto const config_json = oss.str();
    model_builder::PredTransformFunc pred_transform{"exponential_standard_ratio", config_json};

    builder.InitializeMetadata(metadata, tree_annotation, pred_transform, {0.0}, std::nullopt);
  }

  void HandleLeafNode(model_builder::ModelBuilder& builder, int tree_id, int node_id,
      double const** value, [[maybe_unused]] int const* n_classes) const {
    builder.LeafScalar(value[tree_id][node_id]);
  }

 private:
  double ratio_c_;
};

class GradientBoostingRegressorMixIn {
 public:
  explicit GradientBoostingRegressorMixIn(double base_score) : base_score_{base_score} {}

  void HandleMetadata(model_builder::ModelBuilder& builder, int n_trees, int n_features,
      [[maybe_unused]] int n_targets, [[maybe_unused]] std::int32_t const* n_classes) {
    model_builder::Metadata metadata{n_features, TaskType::kRegressor, false, 1, {1}, {1, 1}};
    model_builder::TreeAnnotation tree_annotation{
        n_trees, std::vector<std::int32_t>(n_trees, 0), std::vector<std::int32_t>(n_trees, 0)};
    model_builder::PredTransformFunc pred_transform{"identity"};
    std::vector<double> base_scores{base_score_};
    builder.InitializeMetadata(
        metadata, tree_annotation, pred_transform, base_scores, std::nullopt);
  }

  void HandleLeafNode(model_builder::ModelBuilder& builder, int tree_id, int node_id,
      double const** value, [[maybe_unused]] std::int32_t const* n_classes) const {
    builder.LeafScalar(value[tree_id][node_id]);
  }

 private:
  double base_score_;
};

class GradientBoostingBinaryClassifierMixIn {
 public:
  explicit GradientBoostingBinaryClassifierMixIn(double base_score) : base_score_(base_score) {}

  void HandleMetadata(model_builder::ModelBuilder& builder, int n_trees, int n_features,
      [[maybe_unused]] int n_targets, std::int32_t const* n_classes) {
    model_builder::Metadata metadata{n_features, TaskType::kBinaryClf, false, 1, {1}, {1, 1}};
    std::vector<std::int32_t> target_id(n_trees, 0);
    std::vector<std::int32_t> class_id(n_trees, 0);
    model_builder::TreeAnnotation tree_annotation{n_trees, target_id, class_id};
    model_builder::PredTransformFunc pred_transform{"sigmoid"};
    builder.InitializeMetadata(
        metadata, tree_annotation, pred_transform, {base_score_}, std::nullopt);
  }

  void HandleLeafNode(model_builder::ModelBuilder& builder, int tree_id, int node_id,
      double const** value, [[maybe_unused]] std::int32_t const* n_classes) const {
    builder.LeafScalar(value[tree_id][node_id]);
  }

 private:
  double base_score_;
};

class GradientBoostingMulticlassClassifierMixIn {
 public:
  explicit GradientBoostingMulticlassClassifierMixIn(std::vector<double> const& base_scores)
      : base_scores_(base_scores) {}

  void HandleMetadata(model_builder::ModelBuilder& builder, int n_trees, int n_features,
      [[maybe_unused]] int n_targets, std::int32_t const* n_classes) {
    model_builder::Metadata metadata{
        n_features, TaskType::kMultiClf, false, 1, {n_classes[0]}, {1, 1}};
    std::vector<std::int32_t> target_id(n_trees, 0);
    std::vector<std::int32_t> class_id(n_trees);
    for (int tree_id = 0; tree_id < n_trees; ++tree_id) {
      class_id[tree_id] = tree_id % n_classes[0];
    }
    model_builder::TreeAnnotation tree_annotation{n_trees, target_id, class_id};
    model_builder::PredTransformFunc pred_transform{"softmax"};
    builder.InitializeMetadata(
        metadata, tree_annotation, pred_transform, base_scores_, std::nullopt);
  }

  void HandleLeafNode(model_builder::ModelBuilder& builder, int tree_id, int node_id,
      double const** value, [[maybe_unused]] std::int32_t const* n_classes) const {
    builder.LeafScalar(value[tree_id][node_id]);
  }

 private:
  std::vector<double> base_scores_;
};

class HistGradientBoostingRegressorMixIn {
 public:
  explicit HistGradientBoostingRegressorMixIn(double base_score) : base_score_{base_score} {}

  void HandleMetadata(model_builder::ModelBuilder& builder, int n_trees, int n_features,
      [[maybe_unused]] int n_targets, [[maybe_unused]] std::int32_t const* n_classes) {
    model_builder::Metadata metadata{n_features, TaskType::kRegressor, false, 1, {1}, {1, 1}};
    model_builder::TreeAnnotation tree_annotation{
        n_trees, std::vector<std::int32_t>(n_trees, 0), std::vector<std::int32_t>(n_trees, 0)};
    model_builder::PredTransformFunc pred_transform{"identity"};
    std::vector<double> base_scores{base_score_};
    builder.InitializeMetadata(
        metadata, tree_annotation, pred_transform, base_scores, std::nullopt);
  }

  void HandleLeafNode(model_builder::ModelBuilder& builder, int tree_id, int node_id,
      double const** value, [[maybe_unused]] std::int32_t const* n_classes) const {
    builder.LeafScalar(value[tree_id][node_id]);
  }

 private:
  double base_score_;
};

class HistGradientBoostingBinaryClassifierMixIn {
 public:
  explicit HistGradientBoostingBinaryClassifierMixIn(double base_score) : base_score_{base_score} {}

  void HandleMetadata(model_builder::ModelBuilder& builder, int n_trees, int n_features,
      [[maybe_unused]] int n_targets, [[maybe_unused]] std::int32_t const* n_classes) {
    model_builder::Metadata metadata{n_features, TaskType::kBinaryClf, false, 1, {1}, {1, 1}};
    model_builder::TreeAnnotation tree_annotation{
        n_trees, std::vector<std::int32_t>(n_trees, 0), std::vector<std::int32_t>(n_trees, 0)};
    model_builder::PredTransformFunc pred_transform{"sigmoid"};
    std::vector<double> base_scores{base_score_};
    builder.InitializeMetadata(
        metadata, tree_annotation, pred_transform, base_scores, std::nullopt);
  }

  void HandleLeafNode(model_builder::ModelBuilder& builder, int tree_id, int node_id,
      double const** value, [[maybe_unused]] std::int32_t const* n_classes) const {
    builder.LeafScalar(value[tree_id][node_id]);
  }

 private:
  double base_score_;
};

class HistGradientBoostingMulticlassClassifierMixIn {
 public:
  explicit HistGradientBoostingMulticlassClassifierMixIn(std::vector<double> const& base_scores)
      : base_scores_{base_scores} {}

  void HandleMetadata(model_builder::ModelBuilder& builder, int n_trees, int n_features,
      [[maybe_unused]] int n_targets, std::int32_t const* n_classes) {
    model_builder::Metadata metadata{
        n_features, TaskType::kMultiClf, false, 1, {n_classes[0]}, {1, 1}};
    std::vector<std::int32_t> target_id(n_trees, 0);
    std::vector<std::int32_t> class_id(n_trees);
    for (int tree_id = 0; tree_id < n_trees; ++tree_id) {
      class_id[tree_id] = tree_id % n_classes[0];
    }
    model_builder::TreeAnnotation tree_annotation{n_trees, target_id, class_id};
    model_builder::PredTransformFunc pred_transform{"softmax"};
    std::vector<double> base_scores{base_scores_};
    builder.InitializeMetadata(
        metadata, tree_annotation, pred_transform, base_scores, std::nullopt);
  }

  void HandleLeafNode(model_builder::ModelBuilder& builder, int tree_id, int node_id,
      double const** value, [[maybe_unused]] std::int32_t const* n_classes) const {
    builder.LeafScalar(value[tree_id][node_id]);
  }

 private:
  std::vector<double> base_scores_;
};

template <typename MixIn>
std::unique_ptr<Model> LoadSKLearnModel(MixIn& mixin, int n_trees, int n_features, int n_targets,
    std::int32_t const* n_classes, std::int64_t const* node_count,
    std::int64_t const** children_left, std::int64_t const** children_right,
    std::int64_t const** feature, double const** threshold, double const** value,
    std::int64_t const** n_node_samples, double const** weighted_n_node_samples,
    double const** impurity) {
  TREELITE_CHECK_GT(n_trees, 0) << "n_trees must be at least 1";
  TREELITE_CHECK_GT(n_features, 0) << "n_features must be at least 1";

  std::unique_ptr<model_builder::ModelBuilder> builder
      = model_builder::GetModelBuilder(TypeInfo::kFloat64, TypeInfo::kFloat64);
  mixin.HandleMetadata(*builder, n_trees, n_features, n_targets, n_classes);

  for (int tree_id = 0; tree_id < n_trees; ++tree_id) {
    std::int64_t const total_sample_cnt = n_node_samples[tree_id][0];
    TREELITE_CHECK_LE(
        node_count[tree_id], static_cast<std::int64_t>(std::numeric_limits<int>::max()))
        << "Too many nodes in the tree";
    int const n_nodes = static_cast<int>(node_count[tree_id]);

    builder->StartTree();
    for (int node_id = 0; node_id < n_nodes; ++node_id) {
      int const left_child_id = static_cast<int>(children_left[tree_id][node_id]);
      int const right_child_id = static_cast<int>(children_right[tree_id][node_id]);
      std::int64_t sample_cnt = n_node_samples[tree_id][node_id];
      double const weighted_sample_cnt = weighted_n_node_samples[tree_id][node_id];

      builder->StartNode(node_id);
      if (left_child_id == -1) {  // leaf node
        mixin.HandleLeafNode(*builder, tree_id, node_id, value, n_classes);
      } else {
        std::int64_t const split_index = feature[tree_id][node_id];
        double const split_cond = threshold[tree_id][node_id];
        std::int64_t const left_child_sample_cnt = n_node_samples[tree_id][left_child_id];
        std::int64_t const right_child_sample_cnt = n_node_samples[tree_id][right_child_id];
        double const gain
            = static_cast<double>(sample_cnt)
              * (impurity[tree_id][node_id]
                  - static_cast<double>(left_child_sample_cnt) * impurity[tree_id][left_child_id]
                        / static_cast<double>(sample_cnt)
                  - static_cast<double>(right_child_sample_cnt) * impurity[tree_id][right_child_id]
                        / static_cast<double>(sample_cnt))
              / static_cast<double>(total_sample_cnt);

        TREELITE_CHECK_LE(split_index, std::numeric_limits<std::int32_t>::max())
            << "split_index too large";
        builder->NumericalTest(static_cast<std::int32_t>(split_index), split_cond, true,
            Operator::kLE, left_child_id, right_child_id);
        builder->Gain(gain);
      }
      builder->DataCount(sample_cnt);
      builder->SumHess(weighted_sample_cnt);
      builder->EndNode();
    }
    builder->EndTree();
  }
  return builder->CommitModel();
}

template <typename MixIn>
std::unique_ptr<treelite::Model> LoadHistGradientBoosting(MixIn& mixin, int n_trees, int n_features,
    std::int32_t n_classes, std::int64_t const* node_count, std::int64_t const** children_left,
    std::int64_t const** children_right, std::int64_t const** feature, double const** threshold,
    std::int8_t const** default_left, double const** value, std::int64_t const** n_node_samples,
    double const** gain) {
  TREELITE_CHECK_GT(n_trees, 0) << "n_trees must be at least 1";
  TREELITE_CHECK_GT(n_features, 0) << "n_features must be at least 1";

  std::unique_ptr<model_builder::ModelBuilder> builder
      = model_builder::GetModelBuilder(TypeInfo::kFloat64, TypeInfo::kFloat64);
  mixin.HandleMetadata(*builder, n_trees, n_features, 1, &n_classes);

  for (int tree_id = 0; tree_id < n_trees; ++tree_id) {
    TREELITE_CHECK_LE(
        node_count[tree_id], static_cast<std::int64_t>(std::numeric_limits<int>::max()))
        << "Too many nodes in the tree";
    int const n_nodes = static_cast<int>(node_count[tree_id]);

    builder->StartTree();
    for (int node_id = 0; node_id < n_nodes; ++node_id) {
      int const left_child_id = static_cast<int>(children_left[tree_id][node_id]);
      int const right_child_id = static_cast<int>(children_right[tree_id][node_id]);
      std::int64_t const sample_cnt = n_node_samples[tree_id][node_id];
      builder->StartNode(node_id);
      if (left_child_id == -1) {  // leaf node
        mixin.HandleLeafNode(*builder, tree_id, node_id, value, &n_classes);
      } else {
        const std::int64_t split_index = feature[tree_id][node_id];
        double const split_cond = threshold[tree_id][node_id];
        TREELITE_CHECK_LE(split_index, std::numeric_limits<std::int32_t>::max())
            << "split_index too large";
        builder->NumericalTest(static_cast<std::int32_t>(split_index), split_cond,
            (default_left[tree_id][node_id] == 1), treelite::Operator::kLE, left_child_id,
            right_child_id);
        builder->Gain(gain[tree_id][node_id]);
      }
      builder->DataCount(sample_cnt);
      builder->EndNode();
    }
    builder->EndTree();
  }
  return builder->CommitModel();
}

}  // namespace detail

std::unique_ptr<treelite::Model> LoadRandomForestRegressor(int n_estimators, int n_features,
    int n_targets, std::int64_t const* node_count, std::int64_t const** children_left,
    std::int64_t const** children_right, std::int64_t const** feature, double const** threshold,
    double const** value, std::int64_t const** n_node_samples,
    double const** weighted_n_node_samples, double const** impurity) {
  detail::RandomForestRegressorMixIn mixin{};
  return detail::LoadSKLearnModel(mixin, n_estimators, n_features, n_targets, nullptr, node_count,
      children_left, children_right, feature, threshold, value, n_node_samples,
      weighted_n_node_samples, impurity);
}

std::unique_ptr<treelite::Model> LoadIsolationForest(int n_estimators, int n_features,
    std::int64_t const* node_count, std::int64_t const** children_left,
    std::int64_t const** children_right, std::int64_t const** feature, double const** threshold,
    double const** value, std::int64_t const** n_node_samples,
    double const** weighted_n_node_samples, double const** impurity, double ratio_c) {
  detail::IsolationForestMixIn mixin{ratio_c};
  std::vector<std::int32_t> n_classes{1};
  return detail::LoadSKLearnModel(mixin, n_estimators, n_features, 1, n_classes.data(), node_count,
      children_left, children_right, feature, threshold, value, n_node_samples,
      weighted_n_node_samples, impurity);
}

std::unique_ptr<treelite::Model> LoadRandomForestClassifier(int n_estimators, int n_features,
    int n_targets, int32_t const* n_classes, std::int64_t const* node_count,
    std::int64_t const** children_left, std::int64_t const** children_right,
    std::int64_t const** feature, double const** threshold, double const** value,
    std::int64_t const** n_node_samples, double const** weighted_n_node_samples,
    double const** impurity) {
  detail::RandomForestClassifierMixIn mixin{};
  return detail::LoadSKLearnModel(mixin, n_estimators, n_features, n_targets, n_classes, node_count,
      children_left, children_right, feature, threshold, value, n_node_samples,
      weighted_n_node_samples, impurity);
}

std::unique_ptr<treelite::Model> LoadGradientBoostingRegressor(int n_iter, int n_features,
    std::int64_t const* node_count, std::int64_t const** children_left,
    std::int64_t const** children_right, std::int64_t const** feature, double const** threshold,
    double const** value, std::int64_t const** n_node_samples,
    double const** weighted_n_node_samples, double const** impurity, double const* base_scores) {
  detail::GradientBoostingRegressorMixIn mixin{base_scores[0]};
  return detail::LoadSKLearnModel(mixin, n_iter, n_features, 1, nullptr, node_count, children_left,
      children_right, feature, threshold, value, n_node_samples, weighted_n_node_samples, impurity);
}

std::unique_ptr<treelite::Model> LoadGradientBoostingClassifier(int n_iter, int n_features,
    int n_classes, std::int64_t const* node_count, std::int64_t const** children_left,
    std::int64_t const** children_right, std::int64_t const** feature, double const** threshold,
    double const** value, std::int64_t const** n_node_samples,
    double const** weighted_n_node_samples, double const** impurity, double const* base_scores) {
  TREELITE_CHECK_GE(n_classes, 2) << "Number of classes must be at least 2";
  if (n_classes > 2) {
    std::vector<double> base_scores_(base_scores, base_scores + n_classes);
    detail::GradientBoostingMulticlassClassifierMixIn mixin{base_scores_};
    std::vector<std::int32_t> n_classes_{static_cast<std::int32_t>(n_classes)};
    return detail::LoadSKLearnModel(mixin, n_iter * n_classes, n_features, 1, n_classes_.data(),
        node_count, children_left, children_right, feature, threshold, value, n_node_samples,
        weighted_n_node_samples, impurity);
  } else {
    detail::GradientBoostingBinaryClassifierMixIn mixin{base_scores[0]};
    std::vector<std::int32_t> n_classes_{static_cast<std::int32_t>(n_classes)};
    return detail::LoadSKLearnModel(mixin, n_iter, n_features, 1, n_classes_.data(), node_count,
        children_left, children_right, feature, threshold, value, n_node_samples,
        weighted_n_node_samples, impurity);
  }
}

std::unique_ptr<treelite::Model> LoadHistGradientBoostingRegressor(int n_iter, int n_features,
    std::int64_t const* node_count, std::int64_t const** children_left,
    std::int64_t const** children_right, std::int64_t const** feature, double const** threshold,
    std::int8_t const** default_left, double const** value, std::int64_t const** n_node_samples,
    double const** gain, double const* base_scores) {
  detail::HistGradientBoostingRegressorMixIn mixin{base_scores[0]};
  return detail::LoadHistGradientBoosting(mixin, n_iter, n_features, 1, node_count, children_left,
      children_right, feature, threshold, default_left, value, n_node_samples, gain);
}

std::unique_ptr<treelite::Model> LoadHistGradientBoostingClassifier(int n_iter, int n_features,
    int n_classes, std::int64_t const* node_count, std::int64_t const** children_left,
    std::int64_t const** children_right, std::int64_t const** feature, double const** threshold,
    std::int8_t const** default_left, double const** value, std::int64_t const** n_node_samples,
    double const** gain, double const* base_scores) {
  TREELITE_CHECK_GE(n_classes, 2) << "Number of classes must be at least 2";
  if (n_classes > 2) {
    std::vector<double> base_scores_(base_scores, base_scores + n_classes);
    detail::HistGradientBoostingMulticlassClassifierMixIn mixin{base_scores_};
    return detail::LoadHistGradientBoosting(mixin, n_iter * n_classes, n_features, n_classes,
        node_count, children_left, children_right, feature, threshold, default_left, value,
        n_node_samples, gain);
  } else {
    detail::HistGradientBoostingBinaryClassifierMixIn mixin{base_scores[0]};
    return detail::LoadHistGradientBoosting(mixin, n_iter, n_features, n_classes, node_count,
        children_left, children_right, feature, threshold, default_left, value, n_node_samples,
        gain);
  }
}

}  // namespace treelite::model_loader::sklearn
