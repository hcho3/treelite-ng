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

#include <cfloat>
#include <cmath>
#include <string>
#include <type_traits>
#include <variant>

#include "detail/threading_utils.h"

namespace treelite::gtil {

template <typename ThresholdType>
inline int NextNode(
    float fvalue, ThresholdType threshold, Operator op, int left_child, int right_child) {
  bool cond = false;
  switch (op) {
  case Operator::kLT:
    cond = fvalue < threshold;
    break;
  case Operator::kLE:
    cond = fvalue <= threshold;
    break;
  case Operator::kEQ:
    cond = fvalue == threshold;
    break;
  case Operator::kGT:
    cond = fvalue > threshold;
    break;
  case Operator::kGE:
    cond = fvalue >= threshold;
    break;
  default:
    TREELITE_CHECK(false) << "Unrecognized comparison operator " << static_cast<int>(op);
    return -1;
  }
  return (cond ? left_child : right_child);
}

inline int NextNodeCategorical(float fvalue, std::vector<std::uint32_t> const& category_list,
    bool category_list_right_child, int left_child, int right_child) {
  bool category_matched;
  auto max_representable_int = static_cast<float>(std::uint32_t(1) << FLT_MANT_DIG);
  if (fvalue < 0 || std::fabs(fvalue) > max_representable_int) {
    category_matched = false;
  } else {
    auto const category_value = static_cast<std::uint32_t>(fvalue);
    category_matched = (std::find(category_list.begin(), category_list.end(), category_value)
                        != category_list.end());
  }
  if (category_list_right_child) {
    return category_matched ? right_child : left_child;
  } else {
    return category_matched ? left_child : right_child;
  }
}

template <typename OutputLogic, typename ThresholdType, typename LeafOutputType, typename InputT>
int EvaluateTree(treelite::Tree<ThresholdType, LeafOutputType> const& tree, std::size_t tree_id,
    InputT const* row, std::size_t num_class) {
  int node_id = 0;
  while (!tree.IsLeaf(node_id)) {
    auto const split_index = tree.SplitIndex(node_id);
    if (std::isnan(row[split_index])) {
      node_id = tree.DefaultChild(node_id);
    } else {
      float const fvalue = row[split_index];
      if (tree.SplitType(node_id) == treelite::TreeNodeType::kCategoricalTestNode) {
        node_id = NextNodeCategorical(fvalue, tree.CategoryList(node_id),
            tree.CategoryListRightChild(), tree.LeftChild(node_id), tree.RightChild(node_id));
      } else {
        node_id = NextNode(
            fvalue, tree.Threshold(node_id), tree.ComparisonOp(node_id), tree.LeftChild(node_id));
      }
    }
  }
  return node_id;
}

template <typename InputT>
void PredictRaw(Model const& model, InputT* input, InputT* output,
    threading_utils::ThreadConfig const& config) {}

template <typename InputT>
void PredictLeaf(Model const& model, InputT* input, InputT* output,
    threading_utils::ThreadConfig const& config) {}

template <typename InputT>
void PredictScoreByTree(Model const& model, InputT* input, InputT* output,
    threading_utils::ThreadConfig const& config) {}

template <typename InputT>
void Predict(Model const& model, InputT* input, InputT* output, Configuration const& config) {
  TypeInfo leaf_output_type = model.GetLeafOutputType();
  TypeInfo input_type = TypeInfoFromType<InputT>();
  if (leaf_output_type != input_type) {
    std::string expected = TypeInfoToString(leaf_output_type);
    std::string got = TypeInfoToString(input_type);
    if (got == "invalid") {
      got = typeid(InputT).name();
    }
    TREELITE_LOG(FATAL) << "Incorrect input type passed to GTIL predict(). "
                        << "Expected: " << expected << ", Got: " << got;
  }
  auto thread_config = threading_utils::ConfigureThreadConfig(config.nthread);
  if (config.pred_type == PredictKind::kPredictDefault) {
    PredictRaw(model, input, output, thread_config);
  } else if (config.pred_type == PredictKind::kPredictRaw) {
    PredictRaw(model, input, output, thread_config);
  } else if (config.pred_type == PredictKind::kPredictLeafID) {
    PredictLeaf(model, input, output, thread_config);
  } else if (config.pred_type == PredictKind::kPredictPerTree) {
    PredictScoreByTree(model, input, output, thread_config);
  } else {
    TREELITE_LOG(FATAL) << "Not implemented";
  }
}

template void Predict<float>(Model const&, float*, float*, Configuration const&);
template void Predict<double>(Model const&, double*, double*, Configuration const&);

}  // namespace treelite::gtil
