/*!
 * Copyright (c) 2020-2023 by Contributors
 * \file tree.h
 * \brief Implementation for treelite/tree.h
 * \author Hyunsu Cho
 */
#ifndef TREELITE_DETAIL_TREE_H_
#define TREELITE_DETAIL_TREE_H_

#include <treelite/error.h>
#include <treelite/logging.h>
#include <treelite/version.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

namespace treelite {

template <typename ThresholdType, typename LeafOutputType>
inline Tree<ThresholdType, LeafOutputType> Tree<ThresholdType, LeafOutputType>::Clone() const {
  Tree<ThresholdType, LeafOutputType> tree;
  tree.num_nodes = num_nodes;
  tree.nodes_ = nodes_.Clone();
  tree.leaf_vector_ = leaf_vector_.Clone();
  tree.leaf_vector_begin_ = leaf_vector_begin_.Clone();
  tree.leaf_vector_end_ = leaf_vector_end_.Clone();
  tree.matching_categories_ = matching_categories_.Clone();
  tree.matching_categories_offset_ = matching_categories_offset_.Clone();
  return tree;
}

template <typename ThresholdType, typename LeafOutputType>
inline void Tree<ThresholdType, LeafOutputType>::Node::Init() {
  std::memset(this, 0, sizeof(Node));
  cleft_ = cright_ = -1;
  sindex_ = 0;
  info_.leaf_value = static_cast<LeafOutputType>(0);
  info_.threshold = static_cast<ThresholdType>(0);
  data_count_ = 0;
  sum_hess_ = gain_ = 0.0;
  data_count_present_ = sum_hess_present_ = gain_present_ = false;
  categories_list_right_child_ = false;
  node_type_ = TreeNodeType::kLeafNode;
  cmp_ = Operator::kNone;
}

template <typename ThresholdType, typename LeafOutputType>
inline int Tree<ThresholdType, LeafOutputType>::AllocNode() {
  int nd = num_nodes++;
  if (nodes_.Size() != static_cast<std::size_t>(nd)) {
    throw Error("Invariant violated: nodes_ contains incorrect number of nodes");
  }
  for (int nid = nd; nid < num_nodes; ++nid) {
    leaf_vector_begin_.PushBack(0);
    leaf_vector_end_.PushBack(0);
    matching_categories_offset_.PushBack(matching_categories_offset_.Back());
    nodes_.Resize(nodes_.Size() + 1);
    nodes_.Back().Init();
  }
  return nd;
}

template <typename ThresholdType, typename LeafOutputType>
inline void Tree<ThresholdType, LeafOutputType>::Init() {
  num_nodes = 1;
  has_categorical_split_ = false;
  leaf_vector_.Clear();
  leaf_vector_begin_.Resize(1, {});
  leaf_vector_end_.Resize(1, {});
  matching_categories_.Clear();
  matching_categories_offset_.Resize(2, 0);
  nodes_.Resize(1);
  nodes_.at(0).Init();
  SetLeaf(0, static_cast<LeafOutputType>(0));
}

template <typename ThresholdType, typename LeafOutputType>
inline void Tree<ThresholdType, LeafOutputType>::AddChilds(int nid) {
  int const cleft = this->AllocNode();
  int const cright = this->AllocNode();
  nodes_.at(nid).cleft_ = cleft;
  nodes_.at(nid).cright_ = cright;
}

template <typename ThresholdType, typename LeafOutputType>
inline void Tree<ThresholdType, LeafOutputType>::SetNumericalSplit(
    int nid, unsigned split_index, ThresholdType threshold, bool default_left, Operator cmp) {
  Node& node = nodes_.at(nid);
  if (split_index >= ((1U << 31U) - 1)) {
    throw Error("split_index too big");
  }
  if (default_left) {
    split_index |= (1U << 31U);
  }
  node.sindex_ = split_index;
  (node.info_).threshold = threshold;
  node.cmp_ = cmp;
  node.node_type_ = TreeNodeType::kNumericalTestNode;
  node.categories_list_right_child_ = false;
}

template <typename ThresholdType, typename LeafOutputType>
inline void Tree<ThresholdType, LeafOutputType>::SetCategoricalSplit(
    int nid, unsigned split_index, bool default_left,
    std::vector<std::uint32_t> const& categories_list, bool categories_list_right_child) {
  if (split_index >= ((1U << 31U) - 1)) {
    throw Error("split_index too big");
  }

  const std::size_t end_oft = matching_categories_offset_.Back();
  const std::size_t new_end_oft = end_oft + categories_list.size();
  if (end_oft != matching_categories_.Size()) {
    throw Error("Invariant violated");
  }
  if (!std::all_of(&matching_categories_offset_.at(nid + 1), matching_categories_offset_.End(),
                   [end_oft](std::size_t x) { return (x == end_oft); })) {
    throw Error("Invariant violated");
  }
  // Hopefully we won't have to move any element as we add node_matching_categories for node nid
  matching_categories_.Extend(categories_list);
  if (new_end_oft != matching_categories_.Size()) {
    throw Error("Invariant violated");
  }
  std::for_each(&matching_categories_offset_.at(nid + 1), matching_categories_offset_.End(),
                [new_end_oft](std::size_t& x) { x = new_end_oft; });
  if (!matching_categories_.Empty()) {
    std::sort(&matching_categories_.at(end_oft), matching_categories_.End());
  }

  Node& node = nodes_.at(nid);
  if (default_left) {
    split_index |= (1U << 31U);
  }
  node.sindex_ = split_index;
  node.node_type_ = TreeNodeType::kCategoricalTestNode;
  node.categories_list_right_child_ = categories_list_right_child;

  has_categorical_split_ = true;
}

template <typename ThresholdType, typename LeafOutputType>
inline void Tree<ThresholdType, LeafOutputType>::SetLeaf(int nid, LeafOutputType value) {
  Node& node = nodes_.at(nid);
  (node.info_).leaf_value = value;
  node.cleft_ = -1;
  node.cright_ = -1;
  node.node_type_ = TreeNodeType::kLeafNode;
}

template <typename ThresholdType, typename LeafOutputType>
inline void Tree<ThresholdType, LeafOutputType>::SetLeafVector(
    int nid, std::vector<LeafOutputType> const& node_leaf_vector) {
  std::size_t begin = leaf_vector_.Size();
  std::size_t end = begin + node_leaf_vector.size();
  leaf_vector_.Extend(node_leaf_vector);
  leaf_vector_begin_[nid] = begin;
  leaf_vector_end_[nid] = end;
  Node& node = nodes_.at(nid);
  node.cleft_ = -1;
  node.cright_ = -1;
  node.node_type_ = TreeNodeType::kLeafNode;
}

template <typename ThresholdType, typename LeafOutputType>
inline std::unique_ptr<Model> Model::Create() {
  std::unique_ptr<Model> model = std::make_unique<Model>();
  model->variant_ = ModelPreset<ThresholdType, LeafOutputType>();
  return model;
}

inline std::unique_ptr<Model> Model::Create(TypeInfo threshold_type, TypeInfo leaf_output_type) {
  std::unique_ptr<Model> model = std::make_unique<Model>();
  TREELITE_CHECK(threshold_type == TypeInfo::kFloat32 || threshold_type == TypeInfo::kFloat64);
  TREELITE_CHECK(leaf_output_type == TypeInfo::kUInt32 || leaf_output_type == threshold_type);
  int const target_variant_index
      = (threshold_type == TypeInfo::kFloat64) * 2 + (leaf_output_type == TypeInfo::kUInt32);
  model->variant_ = SetModelPresetVariant<0>(target_variant_index);
  return model;
}

}  // namespace treelite
#endif  // TREELITE_DETAIL_TREE_H_
