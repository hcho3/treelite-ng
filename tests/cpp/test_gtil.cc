/*!
 * Copyright (c) 2023 by Contributors
 * \file test_gtil.cc
 * \author Hyunsu Cho
 * \brief C++ tests for GTIL
 */

#include <fmt/format.h>
#include <gtest/gtest.h>
#include <treelite/enum/operator.h>
#include <treelite/enum/task_type.h>
#include <treelite/enum/typeinfo.h>
#include <treelite/gtil.h>
#include <treelite/model_builder.h>
#include <treelite/tree.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace treelite {

class GTIL : public testing::TestWithParam<std::string> {};

TEST_P(GTIL, MulticlassClfGrovePerClass) {
  model_builder::Metadata metadata{1, TaskType::kMultiClf, false, 1, {3}, {1, 1}};
  model_builder::TreeAnnotation tree_annotation{6, {0, 0, 0, 0, 0, 0}, {0, 1, 2, 0, 1, 2}};
  model_builder::PredTransformFunc pred_transform{"softmax"};
  std::vector<double> base_scores{0.3, 0.2, 0.5};
  std::unique_ptr<model_builder::ModelBuilder> builder
      = model_builder::GetModelBuilder(TypeInfo::kFloat32, TypeInfo::kFloat32, metadata,
          tree_annotation, pred_transform, base_scores);
  auto make_tree_stump = [&](float left_child_val, float right_child_val) {
    builder->StartTree();
    builder->StartNode(0);
    builder->NumericalTest(0, 0.0, false, Operator::kLT, 1, 2);
    builder->EndNode();
    builder->StartNode(1);
    builder->LeafScalar(left_child_val);
    builder->EndNode();
    builder->StartNode(2);
    builder->LeafScalar(right_child_val);
    builder->EndNode();
    builder->EndTree();
  };
  make_tree_stump(-1.0f, 1.0f);
  make_tree_stump(1.0f, -1.0f);
  make_tree_stump(0.5f, 0.5f);
  make_tree_stump(-1.0f, 0.0f);
  make_tree_stump(0.0f, -1.0f);
  make_tree_stump(0.5f, 1.5f);

  auto const predict_kind = GetParam();

  std::unique_ptr<Model> model = builder->CommitModel();
  gtil::Configuration config(fmt::format(R"({{
     "predict_type": "{}",
     "nthread": 1
  }})",
      predict_kind));

  std::vector<std::uint64_t> expected_output_shape;
  std::vector<std::vector<float>> expected_output;
  if (predict_kind == "raw") {
    expected_output_shape = {1, 3};
    expected_output = {{1.3f, -1.8f, 2.5f}, {-1.7f, 1.2f, 1.5f}};
  } else if (predict_kind == "default") {
    expected_output_shape = {1, 3};
    auto softmax = [](float a, float b, float c) {
      float const max = std::max({a, b, c});
      a -= max;
      b -= max;
      c -= max;
      float const sum = std::exp(a) + std::exp(b) + std::exp(c);
      return std::vector<float>{std::exp(a) / sum, std::exp(b) / sum, std::exp(c) / sum};
    };
    expected_output = {softmax(1.3f, -1.8f, 2.5f), softmax(-1.7f, 1.2f, 1.5f)};
  } else if (predict_kind == "leaf_id") {
    expected_output_shape = {1, 6};
    expected_output = {{2, 2, 2, 2, 2, 2}, {1, 1, 1, 1, 1, 1}};
  }
  auto output_shape = gtil::GetOutputShape(*model, 1, config);
  EXPECT_EQ(output_shape, expected_output_shape);

  std::vector<float> output(std::accumulate(
      output_shape.begin(), output_shape.end(), std::uint64_t(1), std::multiplies<>()));
  {
    std::vector<float> input{1.0f};
    gtil::Predict(*model, input.data(), 1, output.data(), config);
    EXPECT_EQ(output, expected_output[0]);
  }
  {
    std::vector<float> input{-1.0f};
    gtil::Predict(*model, input.data(), 1, output.data(), config);
    EXPECT_EQ(output, expected_output[1]);
  }
}

TEST_P(GTIL, LeafVectorRF) {
  model_builder::Metadata metadata{1, TaskType::kMultiClf, true, 1, {3}, {1, 3}};
  model_builder::TreeAnnotation tree_annotation{2, {0, 0}, {-1, -1}};
  model_builder::PredTransformFunc pred_transform{"identity_multiclass"};
  std::vector<double> base_scores{100.0, 200.0, 300.0};
  std::unique_ptr<model_builder::ModelBuilder> builder
      = model_builder::GetModelBuilder(TypeInfo::kFloat32, TypeInfo::kFloat32, metadata,
          tree_annotation, pred_transform, base_scores);
  auto make_tree_stump
      = [&](std::vector<float> const& left_child_val, std::vector<float> const& right_child_val) {
          builder->StartTree();
          builder->StartNode(0);
          builder->NumericalTest(0, 0.0, false, Operator::kLT, 1, 2);
          builder->EndNode();
          builder->StartNode(1);
          builder->LeafVector(left_child_val);
          builder->EndNode();
          builder->StartNode(2);
          builder->LeafVector(right_child_val);
          builder->EndNode();
          builder->EndTree();
        };
  make_tree_stump({1.0f, 0.0f, 0.0f}, {0.0f, 0.5f, 0.5f});
  make_tree_stump({1.0f, 0.0f, 0.0f}, {0.0f, 0.5f, 0.5f});

  auto const predict_kind = GetParam();

  std::unique_ptr<Model> model = builder->CommitModel();
  gtil::Configuration config(fmt::format(R"({{
     "predict_type": "{}",
     "nthread": 1
  }})",
      predict_kind));

  std::vector<std::uint64_t> expected_output_shape;
  std::vector<std::vector<float>> expected_output;
  if (predict_kind == "raw" || predict_kind == "default") {
    expected_output_shape = {1, 3};
    expected_output = {{100.0f, 200.5f, 300.5f}, {101.0f, 200.0f, 300.0f}};
  } else if (predict_kind == "leaf_id") {
    expected_output_shape = {1, 2};
    expected_output = {{2, 2}, {1, 1}};
  }
  auto output_shape = gtil::GetOutputShape(*model, 1, config);
  EXPECT_EQ(output_shape, expected_output_shape);

  std::vector<float> output(std::accumulate(
      output_shape.begin(), output_shape.end(), std::uint64_t(1), std::multiplies<>()));
  {
    std::vector<float> input{1.0f};
    gtil::Predict(*model, input.data(), 1, output.data(), config);
    EXPECT_EQ(output, expected_output[0]);
  }
  {
    std::vector<float> input{-1.0f};
    gtil::Predict(*model, input.data(), 1, output.data(), config);
    EXPECT_EQ(output, expected_output[1]);
  }
}

INSTANTIATE_TEST_SUITE_P(/* no prefix */, GTIL, testing::Values("raw", "default", "leaf_id"));

}  // namespace treelite
