/*!
 * Copyright (c) 2023 by Contributors
 * \file output_size.cc
 * \author Hyunsu Cho
 * \brief Compute output size for GTIL, so that callers can allocate sufficient space
 *        to hold outputs.
 */
#include <treelite/gtil.h>
#include <treelite/tree.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace treelite::gtil {

std::vector<std::uint64_t> GetOutputSize(
    Model const& model, std::uint64_t num_row, Configuration const& config) {
  auto const num_tree = model.GetNumTree();
  auto const max_num_class
      = *std::max_element(model.num_class.Data(), model.num_class.Data() + model.num_target);
  switch (config.pred_type) {
  case PredictKind::kPredictDefault:
  case PredictKind::kPredictRaw:
    if (model.num_target > 1) {
      return {model.num_target, num_row, max_num_class};
    } else {
      return {num_row, max_num_class};
    }
  case PredictKind::kPredictLeafID:
    return {num_row, num_tree};
  case PredictKind::kPredictPerTree:
    return {num_row, num_tree, model.leaf_vector_shape[0] * model.leaf_vector_shape[1]};
  }
}

}  // namespace treelite::gtil
