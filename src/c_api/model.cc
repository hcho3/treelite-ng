/*!
 * Copyright (c) 2023 by Contributors
 * \file model.cc
 * \author Hyunsu Cho
 * \brief C API for functions to query and modify model objects
 */

#include <treelite/c_api.h>
#include <treelite/c_api_error.h>
#include <treelite/tree.h>

#include <string>

#include "./c_api_utils.h"

int TreeliteDumpAsJSON(TreeliteModelHandle handle, int pretty_print, char const** out_json_str) {
  API_BEGIN();
  auto* model_ = static_cast<treelite::Model*>(handle);
  std::string& ret_str = treelite::c_api::ReturnValueStore::Get()->ret_str;
  ret_str = model_->DumpAsJSON(pretty_print != 0);
  *out_json_str = ret_str.c_str();
  API_END();
}

int TreeliteGetInputType(TreeliteModelHandle model, char const** out_str) {
  API_BEGIN();
  auto const* model_ = static_cast<treelite::Model const*>(model);
  auto& type_str = treelite::c_api::ReturnValueStore::Get()->ret_str;
  type_str = treelite::TypeInfoToString(model_->GetThresholdType());
  *out_str = type_str.c_str();
  API_END();
}

int TreeliteGetOutputType(TreeliteModelHandle model, char const** out_str) {
  API_BEGIN();
  auto const* model_ = static_cast<treelite::Model const*>(model);
  auto& type_str = treelite::c_api::ReturnValueStore::Get()->ret_str;
  type_str = treelite::TypeInfoToString(model_->GetLeafOutputType());
  *out_str = type_str.c_str();
  API_END();
}

int TreeliteFreeModel(TreeliteModelHandle handle) {
  API_BEGIN();
  delete static_cast<treelite::Model*>(handle);
  API_END();
}
