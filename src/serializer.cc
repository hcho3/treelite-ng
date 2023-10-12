/*!
 * Copyright (c) 2021-2023 by Contributors
 * \file serializer.cc
 * \brief Implementation for serializers
 * \author Hyunsu Cho
 */

#include <treelite/detail/serializer_mixins.h>
#include <treelite/logging.h>
#include <treelite/tree.h>
#include <treelite/version.h>

#include <iostream>
#include <memory>
#include <variant>

namespace treelite {

namespace detail::serializer {

template <typename MixIn>
class Serializer {
 public:
  explicit Serializer(std::shared_ptr<MixIn> mixin) : mixin_(mixin) {}

  void SerializeHeader(Model& model) {
    // Header 1
    model.major_ver_ = TREELITE_VER_MAJOR;
    model.minor_ver_ = TREELITE_VER_MINOR;
    model.patch_ver_ = TREELITE_VER_PATCH;
    mixin_->SerializeScalar(&model.major_ver_);
    mixin_->SerializeScalar(&model.minor_ver_);
    mixin_->SerializeScalar(&model.patch_ver_);
    model.threshold_type_ = model.GetThresholdType();
    model.leaf_output_type_ = model.GetLeafOutputType();
    mixin_->SerializeScalar(&model.threshold_type_);
    mixin_->SerializeScalar(&model.leaf_output_type_);

    // Number of trees
    model.num_tree_ = static_cast<std::uint64_t>(model.GetNumTree());
    mixin_->SerializeScalar(&model.num_tree_);

    // Header 2
    mixin_->SerializeScalar(&model.num_feature);
    mixin_->SerializeScalar(&model.task_type);
    mixin_->SerializeScalar(&model.average_tree_output);
    mixin_->SerializeScalar(&model.num_target);
    mixin_->SerializeArray(&model.num_class);
    mixin_->SerializeArray(&model.leaf_vector_shape);
    mixin_->SerializeArray(&model.target_id);
    mixin_->SerializeArray(&model.class_id);
    mixin_->SerializeString(&model.postprocessor);
    mixin_->SerializeScalar(&model.sigmoid_alpha);
    mixin_->SerializeScalar(&model.ratio_c);
    mixin_->SerializeArray(&model.base_scores);
    mixin_->SerializeString(&model.attributes);

    /* Extension Slot 1: Per-model optional fields -- to be added later */
    model.num_opt_field_per_model_ = 0;
    mixin_->SerializeScalar(&model.num_opt_field_per_model_);
  }

  void SerializeTrees(Model& model) {
    std::visit(
        [&](auto&& concrete_model) {
          TREELITE_CHECK_EQ(concrete_model.trees.size(), model.num_tree_)
              << "Incorrect number of trees in the model";
          for (auto& tree : concrete_model.trees) {
            SerializeTree(tree);
          }
        },
        model.variant_);
  }

  template <typename ThresholdType, typename LeafOutputType>
  void SerializeTree(Tree<ThresholdType, LeafOutputType>& tree) {
    mixin_->SerializeScalar(&tree.num_nodes);
    mixin_->SerializeScalar(&tree.has_categorical_split_);
    mixin_->SerializeArray(&tree.node_type_);
    mixin_->SerializeArray(&tree.cleft_);
    mixin_->SerializeArray(&tree.cright_);
    mixin_->SerializeArray(&tree.split_index_);
    mixin_->SerializeArray(&tree.default_left_);
    mixin_->SerializeArray(&tree.leaf_value_);
    mixin_->SerializeArray(&tree.threshold_);
    mixin_->SerializeArray(&tree.cmp_);
    mixin_->SerializeArray(&tree.category_list_right_child_);
    mixin_->SerializeArray(&tree.leaf_vector_);
    mixin_->SerializeArray(&tree.leaf_vector_begin_);
    mixin_->SerializeArray(&tree.leaf_vector_end_);
    mixin_->SerializeArray(&tree.category_list_);
    mixin_->SerializeArray(&tree.category_list_begin_);
    mixin_->SerializeArray(&tree.category_list_end_);

    // Node statistics
    mixin_->SerializeArray(&tree.data_count_);
    mixin_->SerializeArray(&tree.data_count_present_);
    mixin_->SerializeArray(&tree.sum_hess_);
    mixin_->SerializeArray(&tree.sum_hess_present_);
    mixin_->SerializeArray(&tree.gain_);
    mixin_->SerializeArray(&tree.gain_present_);

    /* Extension slot 2: Per-tree optional fields -- to be added later */
    tree.num_opt_field_per_tree_ = 0;
    mixin_->SerializeScalar(&tree.num_opt_field_per_tree_);

    /* Extension slot 3: Per-node optional fields -- to be added later */
    tree.num_opt_field_per_node_ = 0;
    mixin_->SerializeScalar(&tree.num_opt_field_per_node_);
  }

 private:
  std::shared_ptr<MixIn> mixin_;
};

template <typename MixIn>
class Deserializer {
 public:
  explicit Deserializer(std::shared_ptr<MixIn> mixin) : mixin_(mixin) {}

  std::unique_ptr<Model> DeserializeHeaderAndCreateModel() {
    // Header 1
    std::int32_t major_ver, minor_ver, patch_ver;
    mixin_->DeserializeScalar(&major_ver);
    mixin_->DeserializeScalar(&minor_ver);
    mixin_->DeserializeScalar(&patch_ver);
    if (major_ver != TREELITE_VER_MAJOR && !(major_ver == 3 && minor_ver == 9)) {
      TREELITE_LOG(FATAL) << "Cannot load model from a different major Treelite version or "
                          << "a version before 3.9.0." << std::endl
                          << "Currently running Treelite version " << TREELITE_VER_MAJOR << "."
                          << TREELITE_VER_MINOR << "." << TREELITE_VER_PATCH << std::endl
                          << "The model checkpoint was generated from Treelite version "
                          << major_ver << "." << minor_ver << "." << patch_ver;
    } else if (major_ver == TREELITE_VER_MAJOR && minor_ver > TREELITE_VER_MINOR) {
      TREELITE_LOG(WARNING)
          << "The model you are loading originated from a newer Treelite version; some "
          << "functionalities may be unavailable." << std::endl
          << "Currently running Treelite version " << TREELITE_VER_MAJOR << "."
          << TREELITE_VER_MINOR << "." << TREELITE_VER_PATCH << std::endl
          << "The model checkpoint was generated from Treelite version " << major_ver << "."
          << minor_ver << "." << patch_ver;
    }
    TypeInfo threshold_type, leaf_output_type;
    mixin_->DeserializeScalar(&threshold_type);
    mixin_->DeserializeScalar(&leaf_output_type);

    std::unique_ptr<Model> model = Model::Create(threshold_type, leaf_output_type);
    model->major_ver_ = major_ver;
    model->minor_ver_ = minor_ver;
    model->patch_ver_ = patch_ver;

    // Number of trees
    mixin_->DeserializeScalar(&model->num_tree_);

    // Header 2
    mixin_->DeserializeScalar(&model->num_feature);
    mixin_->DeserializeScalar(&model->task_type);
    mixin_->DeserializeScalar(&model->average_tree_output);
    mixin_->DeserializeScalar(&model->num_target);
    mixin_->DeserializeArray(&model->num_class);
    mixin_->DeserializeArray(&model->leaf_vector_shape);
    mixin_->DeserializeArray(&model->target_id);
    mixin_->DeserializeArray(&model->class_id);
    mixin_->DeserializeString(&model->postprocessor);
    mixin_->DeserializeScalar(&model->sigmoid_alpha);
    mixin_->DeserializeScalar(&model->ratio_c);
    mixin_->DeserializeArray(&model->base_scores);
    mixin_->DeserializeString(&model->attributes);

    /* Extension Slot 1: Per-model optional fields -- to be added later */
    mixin_->DeserializeScalar(&model->num_opt_field_per_model_);
    // Ignore extra fields; the input is likely from a later version of Treelite
    for (std::int32_t i = 0; i < model->num_opt_field_per_model_; ++i) {
      mixin_->SkipOptionalField();
    }

    return model;
  }

  void DeserializeTrees(Model& model) {
    std::visit(
        [&](auto&& concrete_model) {
          concrete_model.trees.clear();
          for (std::uint64_t i = 0; i < model.num_tree_; ++i) {
            concrete_model.trees.emplace_back();
            DeserializeTree(concrete_model.trees.back());
          }
        },
        model.variant_);
  }

  template <typename ThresholdType, typename LeafOutputType>
  void DeserializeTree(Tree<ThresholdType, LeafOutputType>& tree) {
    mixin_->DeserializeScalar(&tree.num_nodes);
    mixin_->DeserializeScalar(&tree.has_categorical_split_);
    mixin_->DeserializeArray(&tree.node_type_);
    mixin_->DeserializeArray(&tree.cleft_);
    mixin_->DeserializeArray(&tree.cright_);
    mixin_->DeserializeArray(&tree.split_index_);
    mixin_->DeserializeArray(&tree.default_left_);
    mixin_->DeserializeArray(&tree.leaf_value_);
    mixin_->DeserializeArray(&tree.threshold_);
    mixin_->DeserializeArray(&tree.cmp_);
    mixin_->DeserializeArray(&tree.category_list_right_child_);
    mixin_->DeserializeArray(&tree.leaf_vector_);
    mixin_->DeserializeArray(&tree.leaf_vector_begin_);
    mixin_->DeserializeArray(&tree.leaf_vector_end_);
    mixin_->DeserializeArray(&tree.category_list_);
    mixin_->DeserializeArray(&tree.category_list_begin_);
    mixin_->DeserializeArray(&tree.category_list_end_);

    // Node statistics
    mixin_->DeserializeArray(&tree.data_count_);
    mixin_->DeserializeArray(&tree.data_count_present_);
    mixin_->DeserializeArray(&tree.sum_hess_);
    mixin_->DeserializeArray(&tree.sum_hess_present_);
    mixin_->DeserializeArray(&tree.gain_);
    mixin_->DeserializeArray(&tree.gain_present_);

    /* Extension slot 2: Per-tree optional fields -- to be added later */
    mixin_->DeserializeScalar(&tree.num_opt_field_per_tree_);
    // Ignore extra fields; the input is likely from a later version of Treelite
    for (std::int32_t i = 0; i < tree.num_opt_field_per_tree_; ++i) {
      mixin_->SkipOptionalField();
    }

    /* Extension slot 3: Per-node optional fields -- to be added later */
    mixin_->DeserializeScalar(&tree.num_opt_field_per_node_);
    // Ignore extra fields; the input is likely from a later version of Treelite
    for (std::int32_t i = 0; i < tree.num_opt_field_per_node_; ++i) {
      mixin_->SkipOptionalField();
    }
  }

 private:
  std::shared_ptr<MixIn> mixin_;
};

}  // namespace detail::serializer

std::vector<PyBufferFrame> Model::SerializeToPyBuffer() {
  auto mixin = std::make_shared<detail::serializer::PyBufferSerializerMixIn>();
  detail::serializer::Serializer<detail::serializer::PyBufferSerializerMixIn> serializer{mixin};
  serializer.SerializeHeader(*this);
  serializer.SerializeTrees(*this);
  return mixin->GetFrames();
}

std::unique_ptr<Model> Model::DeserializeFromPyBuffer(std::vector<PyBufferFrame> const& frames) {
  auto mixin = std::make_shared<detail::serializer::PyBufferDeserializerMixIn>(frames);
  detail::serializer::Deserializer<detail::serializer::PyBufferDeserializerMixIn> deserializer{
      mixin};
  std::unique_ptr<Model> model = deserializer.DeserializeHeaderAndCreateModel();
  deserializer.DeserializeTrees(*model);
  return model;
}

void Model::SerializeToStream(std::ostream& os) {
  auto mixin = std::make_shared<detail::serializer::StreamSerializerMixIn>(os);
  detail::serializer::Serializer<detail::serializer::StreamSerializerMixIn> serializer{mixin};
  serializer.SerializeHeader(*this);
  serializer.SerializeTrees(*this);
}

std::unique_ptr<Model> Model::DeserializeFromStream(std::istream& is) {
  auto mixin = std::make_shared<detail::serializer::StreamDeserializerMixIn>(is);
  detail::serializer::Deserializer<detail::serializer::StreamDeserializerMixIn> deserializer{mixin};
  std::unique_ptr<Model> model = deserializer.DeserializeHeaderAndCreateModel();
  deserializer.DeserializeTrees(*model);
  return model;
}

}  // namespace treelite
