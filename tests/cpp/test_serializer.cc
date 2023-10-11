/*!
 * Copyright (c) 2020 by Contributors
 * \file test_serializer.cc
 * \author Hyunsu Cho
 * \brief C++ tests for model serializer
 */
#include <fmt/format.h>
#include <gtest/gtest.h>
#include <rapidjson/document.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/writer.h>
#include <treelite/enum/task_type.h>
#include <treelite/enum/typeinfo.h>
#include <treelite/error.h>
#include <treelite/model_builder.h>
#include <treelite/tree.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>

using namespace fmt::literals;

namespace {

inline void TestRoundTrip(treelite::Model* model) {
  for (int i = 0; i < 2; ++i) {
    // Test round trip with in-memory serialization
    auto buffer = model->GetPyBuffer();
    std::unique_ptr<treelite::Model> received_model = treelite::Model::CreateFromPyBuffer(buffer);

    // Use ASSERT_TRUE, since ASSERT_EQ will dump all the raw bytes into a string, potentially
    // causing an OOM error
    ASSERT_TRUE(model->DumpAsJSON(false) == received_model->DumpAsJSON(false));
  }

  for (int i = 0; i < 2; ++i) {
    // Test round trip with in-memory serialization (via string)
    std::ostringstream oss;
    oss.exceptions(std::ios::failbit | std::ios::badbit);
    model->SerializeToStream(oss);

    std::istringstream iss(oss.str());
    iss.exceptions(std::ios::failbit | std::ios::badbit);
    std::unique_ptr<treelite::Model> received_model = treelite::Model::DeserializeFromStream(iss);

    // Use ASSERT_TRUE, since ASSERT_EQ will dump all the raw bytes into a string, potentially
    // causing an OOM error
    ASSERT_TRUE(model->DumpAsJSON(false) == received_model->DumpAsJSON(false));
  }

  for (int i = 0; i < 2; ++i) {
    // Test round trip with serialization to a file stream
    std::filesystem::path tmpdir = std::filesystem::temp_directory_path();
    std::filesystem::path filename = tmpdir / (std::string("binary") + std::to_string(i) + ".bin");
    std::unique_ptr<treelite::Model> received_model;
    {
      std::ofstream ofs(filename, std::ios::out | std::ios::binary);
      ASSERT_TRUE(ofs);
      ofs.exceptions(std::ios::failbit | std::ios::badbit);
      model->SerializeToStream(ofs);
    }
    {
      std::ifstream ifs(filename, std::ios::in | std::ios::binary);
      ASSERT_TRUE(ifs);
      ifs.exceptions(std::ios::failbit | std::ios::badbit);
      received_model = treelite::Model::DeserializeFromStream(ifs);
    }

    // Use ASSERT_TRUE, since ASSERT_EQ will dump all the raw bytes into a string, potentially
    // causing an OOM error
    ASSERT_TRUE(model->DumpAsJSON(false) == received_model->DumpAsJSON(false));

    std::filesystem::remove(filename);
  }
}

}  // anonymous namespace

namespace treelite {

template <typename ThresholdType, typename LeafOutputType>
void PyBufferInterfaceRoundTrip_TreeStump() {
  TypeInfo threshold_type = TypeInfoFromType<ThresholdType>();
  TypeInfo leaf_output_type = TypeInfoFromType<LeafOutputType>();
  model_builder::Metadata metadata{2, TaskType::kRegressor, false, 1, {1}, {1, 1}};
  std::unique_ptr<model_builder::ModelBuilder> builder = model_builder::GetModelBuilder(
      threshold_type, leaf_output_type, metadata, model_builder::TreeAnnotation{1, {0}, {0}},
      model_builder::PostProcessorFunc{"identity"}, {0.0});
  builder->StartTree();
  builder->StartNode(0);
  builder->NumericalTest(0, 0.0, true, Operator::kLT, 1, 2);
  builder->EndNode();
  builder->StartNode(1);
  builder->LeafScalar(1.0);
  builder->EndNode();
  builder->StartNode(2);
  builder->LeafScalar(2.0);
  builder->EndNode();
  builder->EndTree();

  std::unique_ptr<Model> model = builder->CommitModel();
  TestRoundTrip(model.get());

  /* Test correctness of JSON dump */
  std::string expected_json_dump_str = fmt::format(R"JSON(
  {{
    "num_feature": 2,
    "task_type": "kRegressor",
    "average_tree_output": false,
    "num_target": 1,
    "num_class": [1],
    "leaf_vector_shape": [1, 1],
    "target_id": [0],
    "class_id": [0],
    "postprocessor": "identity",
    "sigmoid_alpha": 1.0,
    "ratio_c": 1.0,
    "base_scores": [0.0],
    "attributes": "{{}}",
    "trees": [{{
            "num_nodes": 3,
            "has_categorical_split": false,
            "nodes": [{{
                    "node_id": 0,
                    "split_feature_id": 0,
                    "default_left": true,
                    "node_type": "numerical_test_node",
                    "comparison_op": "<",
                    "threshold": {threshold},
                    "left_child": 1,
                    "right_child": 2
                }}, {{
                    "node_id": 1,
                    "leaf_value": {leaf_value0}
                }}, {{
                    "node_id": 2,
                    "leaf_value": {leaf_value1}
                }}]
        }}]
  }}
  )JSON",
      "threshold"_a = static_cast<ThresholdType>(0),
      "leaf_value0"_a = static_cast<LeafOutputType>(1),
      "leaf_value1"_a = static_cast<LeafOutputType>(2));

  rapidjson::Document json_dump;
  json_dump.Parse(model->DumpAsJSON(false).c_str());

  rapidjson::Document expected_json_dump;
  expected_json_dump.Parse(expected_json_dump_str.c_str());
  EXPECT_TRUE(json_dump == expected_json_dump);
}

TEST(PyBufferInterfaceRoundTrip, TreeStump) {
  PyBufferInterfaceRoundTrip_TreeStump<float, float>();
  PyBufferInterfaceRoundTrip_TreeStump<double, double>();
  ASSERT_THROW((PyBufferInterfaceRoundTrip_TreeStump<float, double>()), treelite::Error);
  ASSERT_THROW((PyBufferInterfaceRoundTrip_TreeStump<double, float>()), treelite::Error);
  ASSERT_THROW((PyBufferInterfaceRoundTrip_TreeStump<std::uint32_t, float>()), treelite::Error);
  ASSERT_THROW((PyBufferInterfaceRoundTrip_TreeStump<std::uint32_t, double>()), treelite::Error);
}

template <typename ThresholdType, typename LeafOutputType>
void PyBufferInterfaceRoundTrip_TreeStumpLeafVec() {
  TypeInfo threshold_type = TypeInfoFromType<ThresholdType>();
  TypeInfo leaf_output_type = TypeInfoFromType<LeafOutputType>();
  model_builder::Metadata metadata{2, TaskType::kMultiClf, true, 1, {2}, {1, 2}};
  model_builder::TreeAnnotation tree_annotation{1, {0}, {-1}};
  std::unique_ptr<model_builder::ModelBuilder> builder
      = model_builder::GetModelBuilder(threshold_type, leaf_output_type, metadata, tree_annotation,
          model_builder::PostProcessorFunc{"identity"}, {0.0, 0.0});
  builder->StartTree();
  builder->StartNode(0);
  builder->NumericalTest(0, 0.0, true, Operator::kLT, 1, 2);
  builder->EndNode();
  builder->StartNode(1);
  builder->LeafVector(
      std::vector<LeafOutputType>{static_cast<LeafOutputType>(1), static_cast<LeafOutputType>(2)});
  builder->EndNode();
  builder->StartNode(2);
  builder->LeafVector(
      std::vector<LeafOutputType>{static_cast<LeafOutputType>(2), static_cast<LeafOutputType>(1)});
  builder->EndNode();
  builder->EndTree();

  std::unique_ptr<Model> model = builder->CommitModel();
  TestRoundTrip(model.get());

  /* Test correctness of JSON dump */
  std::string expected_json_dump_str = fmt::format(R"JSON(
  {{
    "num_feature": 2,
    "task_type": "kMultiClf",
    "average_tree_output": true,
    "num_target": 1,
    "num_class": [2],
    "leaf_vector_shape": [1, 2],
    "target_id": [0],
    "class_id": [-1],
    "postprocessor": "identity",
    "sigmoid_alpha": 1.0,
    "ratio_c": 1.0,
    "base_scores": [0.0, 0.0],
    "attributes": "{{}}",
    "trees": [{{
            "num_nodes": 3,
            "has_categorical_split": false,
            "nodes": [{{
                    "node_id": 0,
                    "split_feature_id": 0,
                    "default_left": true,
                    "node_type": "numerical_test_node",
                    "comparison_op": "<",
                    "threshold": {threshold},
                    "left_child": 1,
                    "right_child": 2
                }}, {{
                    "node_id": 1,
                    "leaf_value": [{leaf_value0}, {leaf_value1}]
                }}, {{
                    "node_id": 2,
                    "leaf_value": [{leaf_value2}, {leaf_value3}]
                }}]
        }}]
  }}
  )JSON",
      "threshold"_a = static_cast<ThresholdType>(0),
      "leaf_value0"_a = static_cast<LeafOutputType>(1),
      "leaf_value1"_a = static_cast<LeafOutputType>(2),
      "leaf_value2"_a = static_cast<LeafOutputType>(2),
      "leaf_value3"_a = static_cast<LeafOutputType>(1));
  rapidjson::Document json_dump;
  json_dump.Parse(model->DumpAsJSON(false).c_str());

  rapidjson::Document expected_json_dump;
  expected_json_dump.Parse(expected_json_dump_str.c_str());
  EXPECT_TRUE(json_dump == expected_json_dump);
}

TEST(PyBufferInterfaceRoundTrip, TreeStumpLeafVec) {
  PyBufferInterfaceRoundTrip_TreeStumpLeafVec<float, float>();
  PyBufferInterfaceRoundTrip_TreeStumpLeafVec<double, double>();
  ASSERT_THROW((PyBufferInterfaceRoundTrip_TreeStumpLeafVec<float, double>()), treelite::Error);
  ASSERT_THROW((PyBufferInterfaceRoundTrip_TreeStumpLeafVec<double, float>()), treelite::Error);
}

}  // namespace treelite
