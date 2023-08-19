/*!
 * Copyright (c) 2017-2023 by Contributors
 * \file tree.h
 * \brief model structure for tree ensemble
 * \author Hyunsu Cho
 */
#ifndef TREELITE_TREE_H_
#define TREELITE_TREE_H_

#include <treelite/contiguous_array.h>
#include <treelite/logging.h>
#include <treelite/pybuffer_frame.h>
#include <treelite/enum/operator.h>
#include <treelite/enum/task_type.h>
#include <treelite/enum/tree_node_type.h>
#include <treelite/enum/typeinfo.h>
#include <treelite/version.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#define __TREELITE_STR(x) #x
#define _TREELITE_STR(x) __TREELITE_STR(x)

#define TREELITE_MAX_PRED_TRANSFORM_LENGTH 256

/* Indicator that certain functions should be visible from a library (Windows only) */
#if defined(_MSC_VER) || defined(_WIN32)
#define TREELITE_DLL_EXPORT __declspec(dllexport)
#else
#define TREELITE_DLL_EXPORT
#endif

namespace treelite::detail::serializer {

template <typename MixIn>
class Serializer;
template <typename MixIn>
class Deserializer;

}  // namespace treelite::detail::serializer

namespace treelite {

class GTILBridge;

// Used for returning version triple from a Model object
struct Version {
  std::int32_t major_ver;
  std::int32_t minor_ver;
  std::int32_t patch_ver;
};

/*! \brief in-memory representation of a decision tree */
template <typename ThresholdType, typename LeafOutputType>
class Tree {
 public:
  static_assert(
      std::is_same_v<ThresholdType, float> || std::is_same_v<ThresholdType, double>,
      "ThresholdType must be either float32 or float64");
  static_assert(std::is_same_v<LeafOutputType, float> || std::is_same_v<LeafOutputType, double>,
      "LeafOutputType must be one of uint32_t, float32 or float64");
  static_assert(std::is_same_v<ThresholdType, LeafOutputType>,
      "Unsupported combination of ThresholdType and LeafOutputType");

  Tree() = default;
  ~Tree() = default;
  Tree(Tree const&) = delete;
  Tree& operator=(Tree const&) = delete;
  Tree(Tree&&) noexcept = default;
  Tree& operator=(Tree&&) noexcept = default;

  inline Tree<ThresholdType, LeafOutputType> Clone() const;

 private:
  ContiguousArray<TreeNodeType> node_type_;
  ContiguousArray<std::int32_t> cleft_;
  ContiguousArray<std::int32_t> cright_;
  ContiguousArray<std::int32_t> split_index_;
  ContiguousArray<bool> default_left_;
  ContiguousArray<LeafOutputType> leaf_value_;
  ContiguousArray<ThresholdType> threshold_;
  ContiguousArray<Operator> cmp_;
  ContiguousArray<bool> category_list_right_child_;

  // Leaf vector
  ContiguousArray<LeafOutputType> leaf_vector_;
  ContiguousArray<std::uint64_t> leaf_vector_begin_;
  ContiguousArray<std::uint64_t> leaf_vector_end_;

  // Category list
  ContiguousArray<std::uint32_t> category_list_;
  ContiguousArray<std::uint64_t> category_list_begin_;
  ContiguousArray<std::uint64_t> category_list_end_;

  // Node statistics
  ContiguousArray<std::uint64_t> data_count_;
  ContiguousArray<double> sum_hess_;
  ContiguousArray<double> gain_;
  ContiguousArray<bool> data_count_present_;
  ContiguousArray<bool> sum_hess_present_;
  ContiguousArray<bool> gain_present_;

  bool has_categorical_split_{false};

  /* Note: the following member fields shall be re-computed at serialization time */

  // Number of optional fields in the extension slots
  std::int32_t num_opt_field_per_tree_{0};
  std::int32_t num_opt_field_per_node_{0};

  template <typename WriterType, typename X, typename Y>
  friend void DumpTreeAsJSON(WriterType& writer, Tree<X, Y> const& tree);

  template <typename MixIn>
  friend class detail::serializer::Serializer;
  template <typename MixIn>
  friend class detail::serializer::Deserializer;

  // allocate a new node
  inline int AllocNode();

 public:
  /*! \brief number of nodes */
  std::int32_t num_nodes{0};
  /*! \brief initialize the model with a single root node */
  inline void Init();
  /*!
   * \brief add child nodes to node
   * \param nid node id to add children to
   */
  inline void AddChilds(int nid);

  /** Getters **/
  /*!
   * \brief index of the node's left child
   * \param nid ID of node being queried
   */
  inline int LeftChild(int nid) const {
    return nodes_[nid].LeftChild();
  }
  /*!
   * \brief index of the node's right child
   * \param nid ID of node being queried
   */
  inline int RightChild(int nid) const {
    return nodes_[nid].RightChild();
  }
  /*!
   * \brief index of the node's "default" child, used when feature is missing
   * \param nid ID of node being queried
   */
  inline int DefaultChild(int nid) const {
    return nodes_[nid].DefaultChild();
  }
  /*!
   * \brief feature index of the node's split condition
   * \param nid ID of node being queried
   */
  inline std::uint32_t SplitIndex(int nid) const {
    return nodes_[nid].SplitIndex();
  }
  /*!
   * \brief whether to use the left child node, when the feature in the split condition is missing
   * \param nid ID of node being queried
   */
  inline bool DefaultLeft(int nid) const {
    return nodes_[nid].DefaultLeft();
  }
  /*!
   * \brief whether the node is leaf node
   * \param nid ID of node being queried
   */
  inline bool IsLeaf(int nid) const {
    return nodes_[nid].IsLeaf();
  }
  /*!
   * \brief get leaf value of the leaf node
   * \param nid ID of node being queried
   */
  inline LeafOutputType LeafValue(int nid) const {
    return nodes_[nid].LeafValue();
  }
  /*!
   * \brief get leaf vector of the leaf node; useful for multi-class random forest classifier
   * \param nid ID of node being queried
   */
  inline std::vector<LeafOutputType> LeafVector(int nid) const {
    const std::size_t offset_begin = leaf_vector_begin_[nid];
    const std::size_t offset_end = leaf_vector_end_[nid];
    if (offset_begin >= leaf_vector_.Size() || offset_end > leaf_vector_.Size()) {
      // Return empty vector, to indicate the lack of leaf vector
      return std::vector<LeafOutputType>();
    }
    return std::vector<LeafOutputType>(&leaf_vector_[offset_begin], &leaf_vector_[offset_end]);
    // Use unsafe access here, since we may need to take the address of one past the last
    // element, to follow with the range semantic of std::vector<>.
  }
  /*!
   * \brief tests whether the leaf node has a non-empty leaf vector
   * \param nid ID of node being queried
   */
  inline bool HasLeafVector(int nid) const {
    return leaf_vector_begin_[nid] != leaf_vector_end_[nid];
  }
  /*!
   * \brief get threshold of the node
   * \param nid ID of node being queried
   */
  inline ThresholdType Threshold(int nid) const {
    return nodes_[nid].Threshold();
  }
  /*!
   * \brief get comparison operator
   * \param nid ID of node being queried
   */
  inline Operator ComparisonOp(int nid) const {
    return nodes_[nid].ComparisonOp();
  }
  /*!
   * \brief Get list of all categories belonging to the left/right child node. See the
   *        categories_list_right_child_ field of each split to determine whether this list
   * represents the right child node or the left child node. Categories are integers ranging from 0
   * to (n-1), where n is the number of categories in that particular feature. This list is assumed
   * to be in ascending order. \param nid ID of node being queried
   */
  inline std::vector<std::uint32_t> MatchingCategories(int nid) const {
    const std::size_t offset_begin = matching_categories_offset_[nid];
    const std::size_t offset_end = matching_categories_offset_[nid + 1];
    if (offset_begin >= matching_categories_.Size() || offset_end > matching_categories_.Size()) {
      // Return empty vector, to indicate the lack of any matching categories
      // The node might be a numerical split
      return std::vector<std::uint32_t>();
    }
    return std::vector<std::uint32_t>(
        &matching_categories_[offset_begin], &matching_categories_[offset_end]);
    // Use unsafe access here, since we may need to take the address of one past the last
    // element, to follow with the range semantic of std::vector<>.
  }
  /*!
   * \brief Get the type of a node
   * \param nid ID of node being queried
   */
  inline TreeNodeType NodeType(int nid) const {
    return nodes_[nid].NodeType();
  }
  /*!
   * \brief test whether this node has data count
   * \param nid ID of node being queried
   */
  inline bool HasDataCount(int nid) const {
    return nodes_[nid].HasDataCount();
  }
  /*!
   * \brief get data count
   * \param nid ID of node being queried
   */
  inline std::uint64_t DataCount(int nid) const {
    return nodes_[nid].DataCount();
  }

  /*!
   * \brief test whether this node has hessian sum
   * \param nid ID of node being queried
   */
  inline bool HasSumHess(int nid) const {
    return nodes_[nid].HasSumHess();
  }
  /*!
   * \brief get hessian sum
   * \param nid ID of node being queried
   */
  inline double SumHess(int nid) const {
    return nodes_[nid].SumHess();
  }
  /*!
   * \brief test whether this node has gain value
   * \param nid ID of node being queried
   */
  inline bool HasGain(int nid) const {
    return nodes_[nid].HasGain();
  }
  /*!
   * \brief get gain value
   * \param nid ID of node being queried
   */
  inline double Gain(int nid) const {
    return nodes_[nid].Gain();
  }
  /*!
   * \brief test whether the list given by MatchingCategories(nid) is associated with the right
   *        child node or the left child node
   * \param nid ID of node being queried
   */
  inline bool CategoriesListRightChild(int nid) const {
    return nodes_[nid].CategoriesListRightChild();
  }

  /*!
   * \brief Query whether this tree contains any categorical splits
   */
  inline bool HasCategoricalSplit() const {
    return has_categorical_split_;
  }

  /** Setters **/
  /*!
   * \brief create a numerical split
   * \param nid ID of node being updated
   * \param split_index feature index to split
   * \param threshold threshold value
   * \param default_left the default direction when feature is unknown
   * \param cmp comparison operator to compare between feature value and
   *            threshold
   */
  inline void SetNumericalSplit(
      int nid, unsigned split_index, ThresholdType threshold, bool default_left, Operator cmp);
  /*!
   * \brief create a categorical split
   * \param nid ID of node being updated
   * \param split_index feature index to split
   * \param default_left the default direction when feature is unknown
   * \param categories_list list of categories to belong to either the right child node or the left
   *                        child node. Set categories_list_right_child parameter to indicate
   *                        which node the category list should represent.
   * \param categories_list_right_child whether categories_list indicates the list of categories
   *                                    for the right child node (true) or the left child node
   *                                    (false)
   */
  inline void SetCategoricalSplit(int nid, unsigned split_index, bool default_left,
                                  std::vector<uint32_t> const& categories_list, bool categories_list_right_child);
  /*!
   * \brief set the leaf value of the node
   * \param nid ID of node being updated
   * \param value leaf value
   */
  inline void SetLeaf(int nid, LeafOutputType value);
  /*!
   * \brief set the leaf vector of the node; useful for multi-class random forest classifier
   * \param nid ID of node being updated
   * \param leaf_vector leaf vector
   */
  inline void SetLeafVector(int nid, std::vector<LeafOutputType> const& leaf_vector);
  /*!
   * \brief set the hessian sum of the node
   * \param nid ID of node being updated
   * \param sum_hess hessian sum
   */
  inline void SetSumHess(int nid, double sum_hess) {
    Node& node = nodes_.at(nid);
    node.sum_hess_ = sum_hess;
    node.sum_hess_present_ = true;
  }
  /*!
   * \brief set the data count of the node
   * \param nid ID of node being updated
   * \param data_count data count
   */
  inline void SetDataCount(int nid, uint64_t data_count) {
    Node& node = nodes_.at(nid);
    node.data_count_ = data_count;
    node.data_count_present_ = true;
  }
  /*!
   * \brief set the gain value of the node
   * \param nid ID of node being updated
   * \param gain gain value
   */
  inline void SetGain(int nid, double gain) {
    Node& node = nodes_.at(nid);
    node.gain_ = gain;
    node.gain_present_ = true;
  }
};

/*! \brief Typed portion of the model class */
template <typename ThresholdType, typename LeafOutputType>
class ModelPreset {
 public:
  /*! \brief member trees */
  std::vector<Tree<ThresholdType, LeafOutputType>> trees;

  using threshold_type = ThresholdType;
  using leaf_output_type = LeafOutputType;

  /*! \brief disable copy; use default move */
  ModelPreset() = default;
  ~ModelPreset() = default;
  ModelPreset(ModelPreset const&) = delete;
  ModelPreset& operator=(ModelPreset const&) = delete;
  ModelPreset(ModelPreset&&) noexcept = default;
  ModelPreset& operator=(ModelPreset&&) noexcept = default;

  inline TypeInfo GetThresholdType() const {
    return TypeInfoFromType<ThresholdType>();
  }
  inline TypeInfo GetLeafOutputType() const {
    return TypeInfoFromType<LeafOutputType>();
  }
  inline std::size_t GetNumTree() const {
    return trees.size();
  }
  void SetTreeLimit(std::size_t limit) {
    return trees.resize(limit);
  }
};

using ModelPresetVariant
    = std::variant<ModelPreset<float, float>, ModelPreset<float, std::uint32_t>,
    ModelPreset<double, double>, ModelPreset<double, std::uint32_t>>;

template <int variant_index>
ModelPresetVariant SetModelPresetVariant(int target_variant_index) {
  ModelPresetVariant result;
  if constexpr (variant_index != std::variant_size_v<ModelPresetVariant>) {
    if (variant_index == target_variant_index) {
      using ModelPresetT = std::variant_alternative_t<variant_index, ModelPresetVariant>;
      result = ModelPresetT();
    } else {
      result = SetModelPresetVariant<variant_index + 1>(target_variant_index);
    }
  }
  return result;
}

/*! \brief Model class for tree ensemble model */
class Model {
 public:
  /*! \brief disable copy; use default move */
  Model()
      : major_ver_(TREELITE_VER_MAJOR),
        minor_ver_(TREELITE_VER_MINOR),
        patch_ver_(TREELITE_VER_PATCH) {}
  virtual ~Model() = default;
  Model(Model const&) = delete;
  Model& operator=(Model const&) = delete;
  Model(Model&&) = default;
  Model& operator=(Model&&) = default;

  ModelPresetVariant variant_;

  template <typename ThresholdType, typename LeafOutputType>
  inline static std::unique_ptr<Model> Create();
  inline static std::unique_ptr<Model> Create(TypeInfo threshold_type, TypeInfo leaf_output_type);
  inline TypeInfo GetThresholdType() const {
    return std::visit([](auto&& inner) { return inner.GetThresholdType(); }, variant_);
  }
  inline TypeInfo GetLeafOutputType() const {
    return std::visit([](auto&& inner) { return inner.GetLeafOutputType(); }, variant_);
  }

  inline std::size_t GetNumTree() const {
    return std::visit([](auto&& inner) { return inner.GetNumTree(); }, variant_);
  }
  inline void SetTreeLimit(std::size_t limit) {
    std::visit([=](auto&& inner) { return inner.SetTreeLimit(limit); }, variant_);
  }
  void DumpAsJSON(std::ostream& fo, bool pretty_print) const;

  inline std::string DumpAsJSON(bool pretty_print) const {
    std::ostringstream oss;
    DumpAsJSON(oss, pretty_print);
    return oss.str();
  }

  /* Compatibility Matrix:
     +------------------+----------+----------+----------------+-----------+
     |                  | To: =3.9 | To: =4.0 | To: >=4.1,<5.0 | To: >=5.0 |
     +------------------+----------+----------+----------------+-----------+
     | From: =3.9       | Yes      | Yes      | Yes            | No        |
     | From: =4.0       | No       | Yes      | Yes            | Yes       |
     | From: >=4.1,<5.0 | No       | Yes      | Yes            | Yes       |
     | From: >=5.0      | No       | No       | No             | Yes       |
     +------------------+----------+----------+----------------+-----------+ */

  /* In-memory serialization, zero-copy */
  TREELITE_DLL_EXPORT std::vector<PyBufferFrame> GetPyBuffer();
  TREELITE_DLL_EXPORT static std::unique_ptr<Model> CreateFromPyBuffer(
      std::vector<PyBufferFrame> frames);

  /* Serialization to a file stream */
  void SerializeToStream(std::ostream& os);
  static std::unique_ptr<Model> DeserializeFromStream(std::istream& is);
  /*! \brief Return the Treelite version that produced this Model object. */
  inline Version GetVersion() const {
    return {major_ver_, minor_ver_, patch_ver_};
  }

  /*!
   * \brief number of features used for the model.
   * It is assumed that all feature indices are between 0 and [num_feature]-1.
   */
  std::int32_t num_feature{0};
  /*! \brief Task type */
  TaskType task_type;
  /*! \brief whether to average tree outputs */
  bool average_tree_output{false};

  /* Task parameters */
  std::uint32_t num_target;
  ContiguousArray<std::uint32_t> num_class;
  ContiguousArray<std::uint32_t> leaf_vector_shape;
  /* Per-tree metadata */
  ContiguousArray<std::int32_t> target_id;
  ContiguousArray<std::int32_t> class_id;
  /* Other model parameters */
  ContiguousArray<char> pred_transform;
  float sigmoid_alpha;
  float ratio_c;
  ContiguousArray<double> base_scores;
  ContiguousArray<char> attributes;

 private:
  /* Note: the following member fields shall be re-computed at serialization time */
  // Number of trees
  std::uint64_t num_tree_{0};
  // Number of optional fields in the extension slot
  std::int32_t num_opt_field_per_model_{0};
  // Which Treelite version produced this model
  std::int32_t major_ver_;
  std::int32_t minor_ver_;
  std::int32_t patch_ver_;
  // Type parameters
  TypeInfo threshold_type_{TypeInfo::kInvalid};
  TypeInfo leaf_output_type_{TypeInfo::kInvalid};

  template <typename MixIn>
  friend class detail::serializer::Serializer;
  template <typename MixIn>
  friend class detail::serializer::Deserializer;
};

/*!
 * \brief Concatenate multiple model objects into a single model object by copying
 *        all member trees into the destination model object
 * \param objs List of model objects
 * \return Concatenated model
 */
std::unique_ptr<Model> ConcatenateModelObjects(std::vector<Model const*> const& objs);

}  // namespace treelite

#include <treelite/detail/tree.h>

#endif  // TREELITE_TREE_H_
