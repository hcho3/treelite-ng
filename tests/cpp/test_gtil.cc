/*!
 * Copyright (c) 2023 by Contributors
 * \file test_gtil.cc
 * \author Hyunsu Cho
 * \brief C++ tests for GTIL
 */

#include <gtest/gtest.h>
#include <treelite/enum/operator.h>
#include <treelite/enum/task_type.h>
#include <treelite/enum/tree_node_type.h>
#include <treelite/enum/typeinfo.h>
#include <treelite/model_builder.h>
#include <treelite/tree.h>

#include <memory>

namespace treelite {

TEST(GTIL, MulticlassClfGrovePerClass) {
  model_builder::Metadata metadata{1, TaskType::kMultiClf, false, 1, {3}, {1, 1}};
  model_builder::TreeAnnotation tree_annotation{3, {0, 0, 0}, {0, 1, 2}};
  model_builder::PredTransformFunc pred_transform{"softmax"};
  std::vector<double> base_scores{0.0, 0.0, 0.0};
  std::unique_ptr<model_builder::ModelBuilder> builder
      = model_builder::InitializeModel(TypeInfo::kFloat64, TypeInfo::kFloat64, metadata,
          tree_annotation, pred_transform, base_scores);
  for (int i = 0; i < 3; ++i) {
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
  }

  std::unique_ptr<Model> model = builder->CommitModel();
  TREELITE_LOG(INFO) << model->DumpAsJSON(true);
}

}  // namespace treelite
