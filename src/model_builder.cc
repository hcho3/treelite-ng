/*!
 * Copyright (c) 2023 by Contributors
 * \file model_builder.cc
 * \brief C++ API for constructing Model objects
 * \author Hyunsu Cho
 */
#include <treelite/enum/typeinfo.h>
#include <treelite/model_builder.h>
#include <treelite/tree.h>

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace treelite::model_builder {

std::unique_ptr<ModelBuilder> InitializeModel(TypeInfo threshold_type, TypeInfo leaf_output_type,
    Metadata const& metadata, TreeAnnotation const& tree_annotation,
    PredTransformFunc const& pred_transform, std::vector<double> const& base_scores,
    std::optional<std::string> const& attributes) {
  return {};
}

}  // namespace treelite::model_builder
