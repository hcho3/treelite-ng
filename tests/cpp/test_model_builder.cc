/*!
 * Copyright (c) 2023 by Contributors
 * \file test_model_builder.cc
 * \author Hyunsu Cho
 * \brief C++ tests for GTIL
 */

#include <gtest/gtest.h>
#include <treelite/detail/threading_utils.h>
#include <treelite/enum/operator.h>
#include <treelite/enum/task_type.h>
#include <treelite/enum/typeinfo.h>
#include <treelite/model_builder.h>
#include <treelite/tree.h>

#include <memory>

namespace treelite {

TEST(ModelBuilder, OrphanedNodes) {
  model_builder::Metadata metadata{1, TaskType::kBinaryClf, false, 1, {1}, {1, 1}};
  model_builder::TreeAnnotation tree_annotation{1, {0}, {0}};
  model_builder::PredTransformFunc pred_transform{"softmax"};
  std::vector<double> base_scores{0.0};
  std::unique_ptr<model_builder::ModelBuilder> builder
      = model_builder::InitializeModel(TypeInfo::kFloat32, TypeInfo::kFloat32, metadata,
          tree_annotation, pred_transform, base_scores);
  builder->StartTree();
  builder->StartNode(0);
  builder->LeafScalar(0.0);
  builder->EndNode();
  builder->StartNode(1);
  builder->LeafScalar(1.0);
  builder->EndNode();
  EXPECT_THROW(builder->EndTree(), Error);
}

TEST(ModelBuilder, InvalidNodeID) {
  model_builder::Metadata metadata{1, TaskType::kBinaryClf, false, 1, {1}, {1, 1}};
  model_builder::TreeAnnotation tree_annotation{1, {0}, {0}};
  model_builder::PredTransformFunc pred_transform{"softmax"};
  std::vector<double> base_scores{0.0};
  std::unique_ptr<model_builder::ModelBuilder> builder
      = model_builder::InitializeModel(TypeInfo::kFloat32, TypeInfo::kFloat32, metadata,
          tree_annotation, pred_transform, base_scores);
  builder->StartTree();
  EXPECT_THROW(builder->StartNode(-1), Error);
  builder->StartNode(0);
  EXPECT_THROW(builder->NumericalTest(0, 0.0, true, Operator::kLT, 0, 1), Error);
  EXPECT_THROW(builder->NumericalTest(0, 0.0, true, Operator::kLT, 2, 2), Error);
  EXPECT_THROW(builder->NumericalTest(0, 0.0, true, Operator::kLT, -1, -2), Error);
  EXPECT_THROW(builder->NumericalTest(0, 0.0, true, Operator::kLT, -1, 2), Error);
  EXPECT_THROW(builder->NumericalTest(0, 0.0, true, Operator::kLT, 2, -1), Error);
}

TEST(ModelBuilder, InvalidState) {
  model_builder::Metadata metadata{1, TaskType::kMultiClf, false, 1, {2}, {1, 2}};
  model_builder::TreeAnnotation tree_annotation{1, {0}, {-1}};
  model_builder::PredTransformFunc pred_transform{"identity_multiclass"};
  std::vector<double> base_scores{0.0, 0.0};
  std::unique_ptr<model_builder::ModelBuilder> builder
      = model_builder::InitializeModel(TypeInfo::kFloat32, TypeInfo::kFloat32, metadata,
          tree_annotation, pred_transform, base_scores);
  builder->StartTree();
  EXPECT_THROW(builder->StartTree(), Error);
  EXPECT_THROW(builder->Gain(0.0), Error);
  EXPECT_THROW(builder->NumericalTest(0, 0.0, false, Operator::kLT, 1, 2), Error);
  EXPECT_THROW(builder->EndNode(), Error);
  EXPECT_THROW(builder->EndTree(), Error);  // Cannot have an empty tree with 0 nodes
  EXPECT_THROW(builder->CommitModel(), Error);

  builder->StartNode(0);
  EXPECT_THROW(builder->StartTree(), Error);
  EXPECT_THROW(builder->StartNode(1), Error);
  EXPECT_THROW(builder->EndNode(), Error);  // Cannot have an empty node
  EXPECT_THROW(builder->EndTree(), Error);
  EXPECT_THROW(builder->CommitModel(), Error);

  builder->Gain(0.0);
  builder->NumericalTest(0, 0.0, false, Operator::kLT, 1, 2);
  EXPECT_THROW(builder->StartTree(), Error);
  EXPECT_THROW(builder->StartNode(2), Error);
  EXPECT_THROW(builder->EndTree(), Error);
  EXPECT_THROW(builder->CommitModel(), Error);
  EXPECT_THROW(builder->LeafScalar(0.0), Error);  // Cannot change node kind once specified
  EXPECT_THROW(builder->NumericalTest(0, 0.0, false, Operator::kLT, 1, 2), Error);

  builder->Gain(0.0);
  builder->EndNode();
  EXPECT_THROW(builder->StartTree(), Error);
  EXPECT_THROW(builder->Gain(0.0), Error);
  EXPECT_THROW(builder->LeafVector(std::vector<float>{0.0, 1.0}), Error);
  EXPECT_THROW(builder->EndNode(), Error);
  EXPECT_THROW(builder->CommitModel(), Error);
  EXPECT_THROW(builder->EndTree(), Error);  // Did not yet specify nodes 1 and 2

  builder->StartNode(1);
  EXPECT_THROW(builder->LeafScalar(-1.0), Error);  // Wrong leaf shape
  EXPECT_THROW(builder->LeafVector(std::vector<float>{0.0, 1.0, 2.0}), Error);  // Wrong leaf shape
  builder->LeafVector(std::vector<float>{0.0, 1.0});
  builder->EndNode();

  builder->StartNode(2);
  builder->LeafVector(std::vector<float>{1.0, 0.0});
  builder->EndNode();
  builder->EndTree();
  auto model = builder->CommitModel();
  model->DumpAsJSON(true);

  EXPECT_THROW(builder->StartTree(), Error);
  EXPECT_THROW(builder->StartNode(3), Error);
  EXPECT_THROW(builder->Gain(1.0), Error);
  EXPECT_THROW(builder->LeafVector(std::vector<float>{0.5, 0.5}), Error);
  EXPECT_THROW(builder->EndNode(), Error);
  EXPECT_THROW(builder->EndTree(), Error);
  EXPECT_THROW(builder->CommitModel(), Error);
}

TEST(ModelBuilder, NodeMapping) {
  model_builder::Metadata metadata{1, TaskType::kBinaryClf, false, 1, {1}, {1, 1}};
  model_builder::TreeAnnotation tree_annotation{1, {0}, {0}};
  model_builder::PredTransformFunc pred_transform{"sigmoid"};
  std::vector<double> base_scores{0.0};

  int const n_trial = 10;
  std::vector<std::string> dump(n_trial);
  detail::threading_utils::ThreadConfig config(-1);
  detail::threading_utils::ParallelFor(
      0, n_trial, config, detail::threading_utils::ParallelSchedule::Static(), [&](int i, int) {
        std::unique_ptr<model_builder::ModelBuilder> builder
            = model_builder::InitializeModel(TypeInfo::kFloat64, TypeInfo::kFloat64, metadata,
                tree_annotation, pred_transform, base_scores);
        builder->StartTree();
        builder->StartNode(0 + i * 2);
        builder->NumericalTest(0, 0.0, false, Operator::kLT, 1 + i * 2, 2 + i * 2);
        builder->EndNode();
        builder->StartNode(1 + i * 2);
        builder->LeafScalar(-1.0);
        builder->EndNode();
        builder->StartNode(2 + i * 2);
        builder->LeafScalar(1.0);
        builder->EndNode();
        builder->EndTree();
        std::unique_ptr<Model> model = builder->CommitModel();
        dump[i] = model->DumpAsJSON(true);
      });
  detail::threading_utils::ParallelFor(1, n_trial, config,
      detail::threading_utils::ParallelSchedule::Static(),
      [&](int i, int) { TREELITE_CHECK_EQ(dump[0], dump[i]); });
}

}  // namespace treelite
