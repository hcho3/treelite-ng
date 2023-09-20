/*!
 * Copyright (c) 2021-2023 by Contributors
 * \file sklearn.cc
 * \brief Frontend for scikit-learn models
 * \author Hyunsu Cho
 */
#include <treelite/enum/task_type.h>
#include <treelite/enum/typeinfo.h>
#include <treelite/logging.h>
#include <treelite/model_builder.h>
#include <treelite/model_loader.h>
#include <treelite/tree.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <queue>
#include <tuple>

namespace treelite::model_loader::sklearn {

namespace detail {

class RandomForestRegressorMixIn {
 public:
  void HandleMetadata(model_builder::ModelBuilder& builder, int n_features, int n_targets,
      int const* n_classes) const {
    std::vector<std::int32_t> n_classes_(n_targets);
    std::transform(n_classes, n_classes + n_targets, n_classes_.begin(),
        [](int e) { return static_cast<std::int32_t>(e); });
    auto max_n_classes = *std::max_element(n_classes_.begin(), n_classes_.end());
    model_builder::Metadata metadata{n_features, TaskType::kRegressor, true,
        static_cast<std::int32_t>(n_targets), n_classes_, {n_targets, max_n_classes}};
    model->num_feature = n_features;
    model->average_tree_output = true;
    model->task_type = TaskType::kRegressor;
    model->task_param.grove_per_class = false;
    model->task_param.output_type = treelite::TaskParam::OutputType::kFloat;
    model->task_param.num_class = 1;
    model->task_param.leaf_vector_size = 1;
    std::strncpy(model->param.pred_transform, "identity", sizeof(model->param.pred_transform));
    model->param.global_bias = 0.0f;
  }

  void HandleLeafNode(int tree_id, std::int64_t node_id, int new_node_id, double const** value,
      [[maybe_unused]] int n_classes, treelite::Tree<double, double>& dest_tree) const {
    double const leaf_value = value[tree_id][node_id];
    dest_tree.SetLeaf(new_node_id, leaf_value);
  }
};

template <typename MixIn>
std::unique_ptr<Model> LoadSKLearnModel(MixIn const& mixin, int n_trees, int n_features,
    int n_targets, int const* n_classes, [[maybe_unused]] std::int64_t const* node_count,
    std::int64_t const** children_left, std::int64_t const** children_right,
    std::int64_t const** feature, double const** threshold, double const** value,
    std::int64_t const** n_node_samples, double const** weighted_n_node_samples,
    double const** impurity) {
  TREELITE_CHECK_GT(n_trees, 0) << "n_trees must be at least 1";
  TREELITE_CHECK_GT(n_features, 0) << "n_features must be at least 1";

  std::unique_ptr<model_builder::ModelBuilder> builder
      = model_builder::GetModelBuilder(TypeInfo::kFloat64, TypeInfo::kFloat64);
  mixin.HandleMetadata(*builder, n_features, n_targets, n_classes);

  auto& trees = std::get<ModelPreset<double, double>>(model->variant_).trees;
  for (int tree_id = 0; tree_id < n_trees; ++tree_id) {
    trees.emplace_back();
    treelite::Tree<double, double>& tree = trees.back();
    tree.Init();

    // Assign node ID's so that a breadth-wise traversal would yield
    // the monotonic sequence 0, 1, 2, ...
    std::queue<std::pair<std::int64_t, int>> Q;  // (old ID, new ID) pair
    Q.emplace(0, 0);
    std::int64_t total_sample_cnt = n_node_samples[tree_id][0];
    while (!Q.empty()) {
      std::int64_t node_id;
      int new_node_id;
      std::tie(node_id, new_node_id) = Q.front();
      Q.pop();
      std::int64_t left_child_id = children_left[tree_id][node_id];
      std::int64_t right_child_id = children_right[tree_id][node_id];
      std::int64_t sample_cnt = n_node_samples[tree_id][node_id];
      double const weighted_sample_cnt = weighted_n_node_samples[tree_id][node_id];
      if (left_child_id == -1) {  // leaf node
        mixin.HandleLeafNode(tree_id, node_id, new_node_id, value, n_classes, tree);
      } else {
        std::int64_t split_index = feature[tree_id][node_id];
        double const split_cond = threshold[tree_id][node_id];
        std::int64_t left_child_sample_cnt = n_node_samples[tree_id][left_child_id];
        std::int64_t right_child_sample_cnt = n_node_samples[tree_id][right_child_id];
        double const gain
            = static_cast<double>(sample_cnt)
              * (impurity[tree_id][node_id]
                  - static_cast<double>(left_child_sample_cnt) * impurity[tree_id][left_child_id]
                        / static_cast<double>(sample_cnt)
                  - static_cast<double>(right_child_sample_cnt) * impurity[tree_id][right_child_id]
                        / static_cast<double>(sample_cnt))
              / static_cast<double>(total_sample_cnt);

        tree.AddChilds(new_node_id);
        tree.SetNumericalSplit(new_node_id, split_index, split_cond, true, treelite::Operator::kLE);
        tree.SetGain(new_node_id, gain);
        Q.emplace(left_child_id, tree.LeftChild(new_node_id));
        Q.emplace(right_child_id, tree.RightChild(new_node_id));
      }
      tree.SetDataCount(new_node_id, sample_cnt);
      tree.SetSumHess(new_node_id, weighted_sample_cnt);
    }
  }
  return model;
}

}  // namespace detail

std::unique_ptr<treelite::Model> LoadRandomForestRegressor(int n_estimators, int n_features,
    int n_targets, std::int64_t const* node_count, std::int64_t const** children_left,
    std::int64_t const** children_right, std::int64_t const** feature, double const** threshold,
    double const** value, std::int64_t const** n_node_samples,
    double const** weighted_n_node_samples, double const** impurity) {
  detail::RandomForestRegressorMixIn mixin{};
  std::vector<int> n_classes(n_targets, 1);
  return detail::LoadSKLearnModel(mixin, n_estimators, n_features, n_targets, n_classes.data(),
      node_count, children_left, children_right, feature, threshold, value, n_node_samples,
      weighted_n_node_samples, impurity);
}

}  // namespace treelite::model_loader::sklearn
