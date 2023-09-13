/*!
 * Copyright (c) 2023 by Contributors
 * \file common.h
 * \brief Helper functions for loading models
 * \author Hyunsu Cho
 */

#ifndef SRC_MODEL_LOADER_DETAIL_COMMON_H_
#define SRC_MODEL_LOADER_DETAIL_COMMON_H_

#include <string>

namespace treelite::model_loader::detail {

inline bool StringStartsWith(std::string const& str, std::string const& prefix) {
  return str.rfind(prefix, 0) == 0;
}

}  // namespace treelite::model_loader::detail

#endif  // SRC_MODEL_LOADER_DETAIL_COMMON_H_
