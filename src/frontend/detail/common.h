/*!
 * Copyright (c) 2023 by Contributors
 * \file common.h
 * \brief Helper functions for loading models
 * \author Hyunsu Cho
 */

#ifndef SRC_FRONTEND_DETAIL_COMMON_H_
#define SRC_FRONTEND_DETAIL_COMMON_H_

#include <string>

namespace treelite::frontend::details {

inline bool StringStartsWith(std::string const& str, std::string const& prefix) {
  return str.rfind(prefix, 0) == 0;
}

}  // namespace treelite::frontend::details

#endif  // SRC_FRONTEND_DETAIL_COMMON_H_
