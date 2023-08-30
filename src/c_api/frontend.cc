/*!
 * Copyright (c) 2023 by Contributors
 * \file frontend.cc
 * \author Hyunsu Cho
 * \brief C API for frontend functions
 */

#include <treelite/c_api.h>
#include <treelite/c_api_error.h>
#include <treelite/frontend.h>
#include <treelite/logging.h>
#include <treelite/tree.h>

#include "./c_api_utils.h"

int TreeliteLoadXGBoostModelLegacyBinary(
    char const* filename, [[maybe_unused]] char const* config_json, TreeliteModelHandle* out) {
  // config_json is unused for now
  API_BEGIN();
  std::unique_ptr<treelite::Model> model
      = treelite::frontend::LoadXGBoostModelLegacyBinary(filename);
  *out = static_cast<TreeliteModelHandle>(model.release());
  API_END();
}

int TreeliteLoadXGBoostModelLegacyBinaryFromMemoryBuffer(void const* buf, size_t len,
    [[maybe_unused]] char const* config_json, TreeliteModelHandle* out) {
  // config_json is unused for now
  API_BEGIN();
  std::unique_ptr<treelite::Model> model
      = treelite::frontend::LoadXGBoostModelLegacyBinary(buf, len);
  *out = static_cast<TreeliteModelHandle>(model.release());
  API_END();
}

int TreeliteLoadXGBoostModel(
    char const* filename, char const* config_json, TreeliteModelHandle* out) {
  API_BEGIN();
  std::unique_ptr<treelite::Model> model
      = treelite::frontend::LoadXGBoostModel(filename, config_json);
  *out = static_cast<TreeliteModelHandle>(model.release());
  API_END();
}

int TreeliteLoadXGBoostModelFromString(
    char const* json_str, size_t length, char const* config_json, TreeliteModelHandle* out) {
  API_BEGIN();
  std::unique_ptr<treelite::Model> model
      = treelite::frontend::LoadXGBoostModelFromString(json_str, length, config_json);
  *out = static_cast<TreeliteModelHandle>(model.release());
  API_END();
}
