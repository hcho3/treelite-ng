/*!
 * Copyright (c) 2023 by Contributors
 * \file file_utils.h
 * \brief Helper functions for manipulating files
 * \author Hyunsu Cho
 */

#ifndef SRC_MODEL_LOADER_DETAIL_FILE_UTILS_H_
#define SRC_MODEL_LOADER_DETAIL_FILE_UTILS_H_

#include <treelite/logging.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

namespace treelite::model_loader::detail {

inline std::ifstream OpenFileForReadAsStream(std::string const& filename) {
  auto path = std::filesystem::weakly_canonical(std::filesystem::u8path(filename));
  TREELITE_CHECK(std::filesystem::exists(path)) << "Path " << filename << " does not exist";
  return std::ifstream(path, std::ios::in | std::ios::binary);
}

inline FILE* OpenFileForReadAsFilePtr(std::string const& filename) {
  auto path = std::filesystem::weakly_canonical(std::filesystem::u8path(filename));
  TREELITE_CHECK(std::filesystem::exists(path)) << "Path " << filename << " does not exist";
  FILE* fp;
#ifdef _WIN32
  fp = _wfopen(path.wstring().c_str(), L"rb");
#else
  fp = std::fopen(path.string().c_str(), "rb");
#endif
  TREELITE_CHECK(fp) << "Could not open file " << filename;
  return fp;
}

}  // namespace treelite::model_loader::detail

#endif  // SRC_MODEL_LOADER_DETAIL_FILE_UTILS_H_
