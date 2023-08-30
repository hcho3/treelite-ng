/*!
 * Copyright (c) 2023 by Contributors
 * \file c_api_utils.h
 * \author Hyunsu Cho
 * \brief C API of Treelite, used for interfacing with other languages
 */
#ifndef SRC_C_API_C_API_UTILS_H_
#define SRC_C_API_C_API_UTILS_H_

#include <treelite/thread_local.h>

#include <string>

namespace treelite::c_api {

/*! \brief When returning a complex object from a C API function, we
 *         store the object here and then return a pointer. The
 *         storage is thread-local static storage. */
struct ReturnValueEntry {
  /*! \brief result holder for returning string */
  std::string ret_str;
};
using ReturnValueStore = ThreadLocalStore<ReturnValueEntry>;

}  // namespace treelite::c_api

#endif  // SRC_C_API_C_API_UTILS_H_
