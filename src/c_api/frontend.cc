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

int TreeliteLoadXGBoostModel(char const* filename, TreeliteModelHandle* out) {
  TREELITE_LOG(WARNING) << "TreeliteLoadXGBoostModel() is deprecated. Please use "
                        << "TreeliteLoadXGBoostModelEx() instead.";
  return TreeliteLoadXGBoostModelEx(filename, "{}", out);
}

int TreeliteLoadXGBoostModelEx(
    char const* filename, [[maybe_unused]] char const* config_json, TreeliteModelHandle* out) {
  // config_json is unused for now
  API_BEGIN();
  std::unique_ptr<treelite::Model> model = treelite::frontend::LoadXGBoostModel(filename);
  *out = static_cast<TreeliteModelHandle>(model.release());
  API_END();
}
