/*!
 * Copyright (c) 2023 by Contributors
 * \file model_builder.h
 * \brief C++ API for constructing Model objects
 * \author Hyunsu Cho
 */

#ifndef TREELITE_MODEL_BUILDER_H_
#define TREELITE_MODEL_BUILDER_H_

#include <treelite/enum/operator.h>
#include <treelite/enum/task_type.h>
#include <treelite/enum/tree_node_type.h>
#include <treelite/enum/typeinfo.h>

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace treelite {

class Model;

namespace model_builder {

class Metadata;
class TreeAnnotation;
class PredTransformFunc;

// Note: this object must be accessed by a single thread.
// For parallel tree construction, build multiple model objects and then concatenate them.
class ModelBuilder {
 public:
  virtual void StartTree() = 0;
  virtual void EndTree() = 0;

  virtual void StartNode(int node_key) = 0;
  virtual void EndNode() = 0;

  virtual void NumericalTest(std::int32_t split_index, double threshold, bool default_left,
      Operator cmp, int left_child_key, int right_child_key)
      = 0;
  virtual void CategoricalTest(std::int32_t split_index, bool default_left,
      std::vector<std::uint32_t> const& category_list, bool category_list_right_child,
      int left_child_key, int right_child_key)
      = 0;

  virtual void LeafScalar(double leaf_value) = 0;
  virtual void LeafVector(std::vector<float> const& leaf_vector) = 0;
  virtual void LeafVector(std::vector<double> const& leaf_vector) = 0;

  virtual void Gain(double gain) = 0;
  virtual void DataCount(std::uint64_t data_count) = 0;
  virtual void SumHess(double sum_hess) = 0;

  virtual void InitializeMetadata(Metadata const& metadata, TreeAnnotation const& tree_annotation,
      PredTransformFunc const& pred_transform, std::vector<double> const& base_scores,
      std::optional<std::string> const& attributes)
      = 0;
  virtual std::unique_ptr<Model> CommitModel() = 0;

  virtual ~ModelBuilder() = default;
};

struct TreeAnnotation {
  std::int32_t num_tree{0};
  std::vector<std::int32_t> target_id{};
  std::vector<std::int32_t> class_id{};
  TreeAnnotation(std::int32_t num_tree, std::vector<std::int32_t> const& target_id,
      std::vector<std::int32_t> const& class_id);
};

struct PredTransformFunc {
  std::string name{};
  std::string config_json{};
  explicit PredTransformFunc(
      std::string const& name, std::optional<std::string> config_json = std::nullopt);
};

struct Metadata {
  std::int32_t num_feature{0};
  TaskType task_type{TaskType::kRegressor};
  bool average_tree_output{false};
  std::int32_t num_target{1};
  std::vector<std::int32_t> num_class{1};
  std::array<std::int32_t, 2> leaf_vector_shape{1, 1};
  Metadata(std::int32_t num_feature, TaskType task_type, bool average_tree_output,
      std::int32_t num_target, std::vector<std::int32_t> const& num_class,
      std::array<std::int32_t, 2> const& leaf_vector_shape);
};

std::unique_ptr<ModelBuilder> GetModelBuilder(TypeInfo threshold_type, TypeInfo leaf_output_type,
    Metadata const& metadata, TreeAnnotation const& tree_annotation,
    PredTransformFunc const& pred_transform, std::vector<double> const& base_scores,
    std::optional<std::string> const& attributes = std::nullopt);
std::unique_ptr<ModelBuilder> GetModelBuilder(TypeInfo threshold_type, TypeInfo leaf_output_type);
// Metadata will be provided later

std::unique_ptr<ModelBuilder> GetModelBuilder(std::string const& json_str);
// Initialize metadata from a JSON string

}  // namespace model_builder
}  // namespace treelite

#endif  // TREELITE_MODEL_BUILDER_H_
