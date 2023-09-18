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

#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace treelite {

class GTIL : public testing::TestWithParam<std::string> {};

TEST_P(GTIL, MulticlassClfGrovePerClass) {
  model_builder::Metadata metadata{1, TaskType::kMultiClf, false, 1, {3}, {1, 1}};
  model_builder::TreeAnnotation tree_annotation{6, {0, 0, 0, 0, 0, 0}, {0, 1, 2, 0, 1, 2}};
  model_builder::PredTransformFunc pred_transform{"softmax"};
  std::vector<double> base_scores{0.0, 0.0, 0.0};
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

  std::unique_ptr<Model> model = builder->CommitModel();
  gtil::Configuration config(fmt::format(R"({{
     "predict_type": "{}",
     "nthread": 1
  }})",
      GetParam()));  // TODO(hcho3): Add test for default prediction
  auto output_shape = gtil::GetOutputShape(*model, 1, config);
  auto const expected_output_shape = std::vector<std::uint64_t>{1, 3};
  EXPECT_EQ(output_shape, expected_output_shape);

  std::vector<float> output(3);
  {
    std::vector<float> input{1.0f};
    gtil::Predict(*model, input.data(), 1, output.data(), config);
    std::vector<float> const& expected_output{1.0f, -2.0f, 2.0f};
    EXPECT_EQ(output, expected_output);
  }
  {
    std::vector<float> input{-1.0f};
    gtil::Predict(*model, input.data(), 1, output.data(), config);
    std::vector<float> const& expected_output{-2.0f, 1.0f, 1.0f};
    EXPECT_EQ(output, expected_output);
  }
}

INSTANTIATE_TEST_SUITE_P(/* no prefix */, GTIL, testing::Values("raw"));

}  // namespace treelite
