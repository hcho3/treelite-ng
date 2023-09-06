/*!
 * Copyright (c) 2021-2023 by Contributors
 * \file gtil.h
 * \author Hyunsu Cho
 * \brief General Tree Inference Library (GTIL), providing a reference implementation for
 *        predicting with decision trees.
 */

#ifndef TREELITE_GTIL_H_
#define TREELITE_GTIL_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace treelite {

class Model;

namespace gtil {

/*! \brief Prediction type */
enum class PredictKind : std::int8_t {
  /*!
   * \brief Usual prediction method: sum over trees and apply post-processing.
   * Expected output dimensions:
   * - (num_row, num_class) if num_target == 1
   * - (num_target, num_row, max_num_class) if num_target > 1
   */
  kPredictDefault = 0,
  /*!
   * \brief Sum over trees, but don't apply post-processing; get raw margin scores instead.
   * Expected output dimensions:
   * - (num_row, num_class) if num_target == 1
   * - (num_target, num_row, max_num_class) if num_target > 1
   */
  kPredictRaw = 1,
  /*!
   * \brief Output one (integer) leaf ID per tree.
   * Expected output dimensions: (num_row, num_tree)
   */
  kPredictLeafID = 2,
  /*!
   * \brief Output one or more margin scores per tree.
   * Expected output dimensions:
   * - (num_row, num_tree, num_class) if num_target == 1
   * - (num_row, num_tree, max_num_class) if num_target > 1
   */
  kPredictPerTree = 3
};

/*! \brief Configuration class */
struct Configuration {
  int nthread{0};  // use all threads by default
  PredictKind pred_type{PredictKind::kPredictDefault};
  Configuration() = default;
  explicit Configuration(char const* config_json);
};

template <typename InputT>
void Predict(Model const& model, InputT* input, InputT* output, Configuration const& config);

std::vector<std::uint64_t> GetOutputSize(
    Model const* model, std::uint64_t num_row, Configuration const& config);

extern template void Predict<float>(Model const&, float*, float*, Configuration const&);
extern template void Predict<double>(Model const&, double*, double*, Configuration const&);

}  // namespace gtil
}  // namespace treelite

#endif  // TREELITE_GTIL_H_
