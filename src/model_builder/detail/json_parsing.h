/*!
 * Copyright (c) 2023 by Contributors
 * \file json_parsing.h
 * \brief Helper functions for parsing model spec from JSON
 * \author Hyunsu Cho
 */

#ifndef SRC_MODEL_BUILDER_DETAIL_JSON_PARSING_H_
#define SRC_MODEL_BUILDER_DETAIL_JSON_PARSING_H_

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>
#include <treelite/enum/task_type.h>
#include <treelite/enum/typeinfo.h>
#include <treelite/logging.h>
#include <treelite/model_builder.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace treelite::model_builder::detail::json_parse {

template <typename ValueT>
struct ValueHandler {};

template <>
struct ValueHandler<std::string> {
  template <typename DocumentT>
  static bool Check(DocumentT const& obj) {
    return obj.IsString();
  }

  template <typename DocumentT>
  static std::string Get(DocumentT const& obj) {
    return obj.GetString();
  }
};

template <>
struct ValueHandler<float> {
  template <typename DocumentT>
  static bool Check(DocumentT const& obj) {
    return obj.IsFloat();
  }

  template <typename DocumentT>
  static float Get(DocumentT const& obj) {
    return obj.GetFloat();
  }
};

template <>
struct ValueHandler<double> {
  template <typename DocumentT>
  static bool Check(DocumentT const& obj) {
    return obj.IsDouble();
  }

  template <typename DocumentT>
  static double Get(DocumentT const& obj) {
    return obj.GetDouble();
  }
};

template <>
struct ValueHandler<std::int32_t> {
  template <typename DocumentT>
  static bool Check(DocumentT const& obj) {
    return obj.IsInt();
  }

  template <typename DocumentT>
  static std::int32_t Get(DocumentT const& obj) {
    return static_cast<std::int32_t>(obj.GetInt());
  }
};

template <>
struct ValueHandler<bool> {
  template <typename DocumentT>
  static bool Check(DocumentT const& obj) {
    return obj.IsBool();
  }

  template <typename DocumentT>
  static bool Get(DocumentT const& obj) {
    return obj.GetBool();
  }
};

template <typename ElemT>
struct ValueHandler<std::vector<ElemT>> {
  template <typename DocumentT>
  static bool Check(DocumentT const& obj) {
    return obj.IsArray();
  }

  template <typename DocumentT>
  static std::vector<ElemT> Get(DocumentT const& obj) {
    std::vector<ElemT> result;
    int idx = 0;
    for (auto const& e : obj.GetArray()) {
      TREELITE_CHECK(ValueHandler<ElemT>::Check(e)) << "Unexpected type at index " << idx;
      result.push_back(ValueHandler<ElemT>::Get(e));
      ++idx;
    }
    return result;
  }
};

template <typename ElemT, std::size_t len>
struct ValueHandler<std::array<ElemT, len>> {
  template <typename DocumentT>
  static bool Check(DocumentT const& obj) {
    return obj.IsArray();
  }

  template <typename DocumentT>
  static std::array<ElemT, len> Get(DocumentT const& obj) {
    std::array<ElemT, len> result;
    auto const& array = obj.GetArray();
    TREELITE_CHECK_EQ(array.Size(), len)
        << "Expected an array of length " << len << " but got length " << array.Size();
    int idx = 0;
    for (auto const& e : array) {
      TREELITE_CHECK(ValueHandler<ElemT>::Check(e)) << "Unexpected type at index " << idx;
      result[idx++] = ValueHandler<ElemT>::Get(e);
    }
    return result;
  }
};

template <typename ValueT>
struct ObjectMemberHandler {
  template <typename DocumentT>
  static ValueT Get(DocumentT const& obj, std::string const& field_name,
      std::optional<ValueT> default_value = std::nullopt) {
    TREELITE_CHECK(obj.IsObject()) << "Expected an object";
    auto itr = obj.FindMember(field_name);
    if (itr != obj.MemberEnd() && ValueHandler<ValueT>::Check(itr->value)) {
      return ValueHandler<ValueT>::Get(itr->value);
    } else if (default_value.has_value()) {
      return default_value.value();
    } else {
      TREELITE_LOG(FATAL) << "Field '" << field_name << "' was required but is missing";
      return {};
    }
  }
};

template <typename DocumentT>
auto const& GetMember(DocumentT const& obj, std::string const& field_name) {
  TREELITE_CHECK(obj.IsObject()) << "Expected an object";
  auto itr = obj.FindMember(field_name);
  TREELITE_CHECK(itr != obj.MemberEnd()) << "Field '" << field_name << "' not found";
  return itr->value;
}

template <typename DocumentT>
Metadata ParseMetadata(DocumentT const& obj) {
  TREELITE_CHECK(obj.IsObject()) << "Expected an object";

  auto const& obj_ = GetMember(obj, "metadata");
  return Metadata{ObjectMemberHandler<std::int32_t>::Get(obj_, "num_feature"),
      TaskTypeFromString(ObjectMemberHandler<std::string>::Get(obj_, "task_type")),
      ObjectMemberHandler<bool>::Get(obj_, "average_tree_output"),
      ObjectMemberHandler<std::int32_t>::Get(obj_, "num_target"),
      ObjectMemberHandler<std::vector<std::int32_t>>::Get(obj_, "num_class"),
      ObjectMemberHandler<std::array<std::int32_t, 2>>::Get(obj_, "leaf_vector_shape")};
}

template <typename DocumentT>
TreeAnnotation ParseTreeAnnotation(DocumentT const& obj) {
  auto const& obj_ = GetMember(obj, "tree_annotation");
  return TreeAnnotation{ObjectMemberHandler<std::int32_t>::Get(obj_, "num_tree"),
      ObjectMemberHandler<std::vector<std::int32_t>>::Get(obj_, "target_id"),
      ObjectMemberHandler<std::vector<std::int32_t>>::Get(obj_, "class_id")};
}

template <typename DocumentT>
PredTransformFunc ParsePredTransformFunc(DocumentT const& obj) {
  std::string config_json;
  auto const& obj_ = GetMember(obj, "pred_transform");
  auto itr = obj_.FindMember("config");
  if (itr != obj_.MemberEnd()) {
    TREELITE_CHECK(itr->value.IsObject()) << "Expected an object for field 'config'";
    rapidjson::StringBuffer buffer;
    rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
    itr->value.Accept(writer);
    config_json = std::string(buffer.GetString());
  } else {
    config_json = "{}";  // default to empty JSON object
  }

  return PredTransformFunc{ObjectMemberHandler<std::string>::Get(obj_, "name"), config_json};
}

template <typename DocumentT>
std::optional<std::string> ParseAttributes(DocumentT const& obj) {
  std::optional<std::string> result;
  auto itr = obj.FindMember("attributes");
  if (itr != obj.MemberEnd()) {
    TREELITE_CHECK(itr->value.IsObject()) << "Expected an object for field 'attributes'";
    rapidjson::StringBuffer buffer;
    rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
    itr->value.Accept(writer);
    result = std::string(buffer.GetString());
  } else {
    result = std::nullopt;
  }

  return result;
}

}  // namespace treelite::model_builder::detail::json_parse

#endif  // SRC_MODEL_BUILDER_DETAIL_JSON_PARSING_H_
