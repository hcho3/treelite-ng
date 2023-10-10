/*!
 * Copyright (c) 2023 by Contributors
 * \file common.h
 * \brief Helper functions for loading models
 * \author Hyunsu Cho
 */

#ifndef SRC_MODEL_LOADER_DETAIL_COMMON_H_
#define SRC_MODEL_LOADER_DETAIL_COMMON_H_

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

namespace treelite::model_loader::detail {

inline bool StringStartsWith(std::string const& str, std::string const& prefix) {
  return str.rfind(prefix, 0) == 0;
}

inline std::ifstream OpenFileForReadAsStream(std::string const& filename, bool binary = false) {
  auto path = std::filesystem::weakly_canonical(std::filesystem::u8path(filename));
  TREELITE_CHECK(std::filesystem::exists(path)) << "Path " << filename << " does not exist";
#ifdef _WIN32
  return std::ifstream(path, std::ios::in | std::ios::binary);
#else
  if (binary) {
    return std::ifstream(path, std::ios::in | std::ios::binary);
  } else {
    return std::ifstream(path, std::ios::in);
  }
#endif
}

inline FILE* OpenFileForReadAsFilePtr(std::string const& filename, bool binary = false) {
  auto path = std::filesystem::weakly_canonical(std::filesystem::u8path(filename));
  TREELITE_CHECK(std::filesystem::exists(path)) << "Path " << filename << " does not exist";
  FILE* fp;
#ifdef _WIN32
  fp = _wfopen(path.wstring().c_str(), L"rb");
#else
  if (binary) {
    fp = std::fopen(path.string().c_str(), "rb");
  } else {
    fp = std::fopen(path.string().c_str(), "r");
  }
#endif
  TREELITE_CHECK(fp) << "Could not open file " << filename;
  return fp;
}

}  // namespace treelite::model_loader::detail

#endif  // SRC_MODEL_LOADER_DETAIL_COMMON_H_
