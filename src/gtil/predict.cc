/*!
 * Copyright (c) 2021-2023 by Contributors
 * \file predict.cc
 * \author Hyunsu Cho
 * \brief General Tree Inference Library (GTIL), providing a reference implementation for
 *        predicting with decision trees.
 */
#include <treelite/gtil.h>
#include <treelite/logging.h>
#include <treelite/tree.h>

#include <string>
#include <type_traits>
#include <variant>

#include "detail/threading_utils.h"

using treelite::threading_utils::ThreadConfig;

namespace treelite::gtil {

/*template <typename DMatrixType>
inline std::size_t PredictImpl(treelite::Model const& model, DMatrixType const* input,
    float* output, ThreadConfig const& thread_config,
    treelite::gtil::Configuration const& pred_config, std::vector<std::size_t>& output_shape) {
  using treelite::gtil::PredictKind;
  if (pred_config.pred_type == PredictKind::kPredictDefault) {
    PredictRaw(model, input, output, thread_config);
    return PredTransform(model, input, output, thread_config, pred_config, output_shape);
  } else if (pred_config.pred_type == PredictKind::kPredictRaw) {
    PredictRaw(model, input, output, thread_config);
    output_shape = {input->GetNumRow(), model.task_param.num_class};
    return input->GetNumRow() * model.task_param.num_class;
  } else if (pred_config.pred_type == PredictKind::kPredictLeafID) {
    PredictLeaf(model, input, output, thread_config);
    output_shape = {input->GetNumRow(), model.GetNumTree()};
    return input->GetNumRow() * model.GetNumTree();
  } else if (pred_config.pred_type == PredictKind::kPredictPerTree) {
    return PredictScoreByTree(model, input, output, thread_config, output_shape);
  } else {
    TREELITE_LOG(FATAL) << "Not implemented";
    return 0;
  }
}*/

template <typename InputT>
void Predict(Model const& model, InputT* input, InputT* output, Configuration const& config) {
  std::visit(
      [](auto&& concrete_model) {
        using ModelType = std::remove_const_t<std::remove_reference_t<decltype(concrete_model)>>;
        using LeafOutputType = typename ModelType::leaf_output_type;
        if constexpr (std::is_same_v<LeafOutputType, InputT>) {
        } else {
          std::string expected = TypeInfoToString(TypeInfoFromType<LeafOutputType>());
          std::string got = TypeInfoToString(TypeInfoFromType<InputT>());
          if (got == "invalid") {
            got = typeid(InputT).name();
          }
          TREELITE_LOG(FATAL) << "Incorrect input type passed to GTIL predict(). "
                              << "Expected: " << expected << ", Got: " << got;
        }
      },
      model.variant_);
}

template void Predict<float>(Model const&, float*, float*, Configuration const&);
template void Predict<double>(Model const&, double*, double*, Configuration const&);

}  // namespace treelite::gtil
