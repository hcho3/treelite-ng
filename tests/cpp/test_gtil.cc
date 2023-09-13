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
  std::unique_ptr<model_builder::ModelBuilder> model
      = model_builder::InitializeModel(TypeInfo::kFloat32, TypeInfo::kFloat32, metadata,
          tree_annotation, pred_transform, base_scores);
}

}  // namespace treelite
