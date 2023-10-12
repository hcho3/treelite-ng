/*!
 * Copyright (c) 2023 by Contributors
 * \file test_model_loader.cc
 * \author Hyunsu Cho
 * \brief C++ tests for model loader
 */

#include <gtest/gtest.h>
#include <model_loader/detail/file_utils.h>
#include <model_loader/detail/string_utils.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

TEST(ModelLoader, StringTrim) {
  std::string s{"foobar\r\n"};
  treelite::model_loader::detail::StringTrimFromEnd(s);
  EXPECT_EQ(s, "foobar");
}

TEST(ModelLoader, StringStartsWith) {
  std::string s{"foobar"};
  EXPECT_TRUE(treelite::model_loader::detail::StringStartsWith(s, "foo"));
}

TEST(ModelLoader, OpenFileForReadAsStream) {
  std::string s{"Hello world"};
  std::string s2;
  std::filesystem::path tmpdir = std::filesystem::temp_directory_path();
  std::filesystem::path filepath = tmpdir / std::filesystem::u8path("ななひら.txt");

  {
    std::ofstream ofs(filepath, std::ios::out | std::ios::binary);
    ASSERT_TRUE(ofs);
    ofs.exceptions(std::ios::failbit | std::ios::badbit);
    ofs.write(s.data(), s.length());
  }
  {
    std::ifstream ifs
        = treelite::model_loader::detail::OpenFileForReadAsStream(filepath.u8string());
    ASSERT_TRUE(ifs);
    ifs.exceptions(std::ios::failbit | std::ios::badbit);
    s2.resize(s.length());
    ifs.read(s2.data(), s.length());
    ASSERT_EQ(s, s2);
  }

  std::filesystem::remove(filepath);
}

TEST(ModelLoader, OpenFileForReadAsFilePtr) {
  std::string s{"Hello world"};
  std::string s2;
  std::filesystem::path tmpdir = std::filesystem::temp_directory_path();
  std::filesystem::path filepath = tmpdir / std::filesystem::u8path("ななひら.txt");

  {
    std::ofstream ofs(filepath, std::ios::out | std::ios::binary);
    ASSERT_TRUE(ofs);
    ofs.exceptions(std::ios::failbit | std::ios::badbit);
    ofs.write(s.data(), s.length());
  }
  {
    FILE* fp = treelite::model_loader::detail::OpenFileForReadAsFilePtr(filepath.u8string());
    ASSERT_TRUE(fp);
    s2.resize(s.length());
    ASSERT_EQ(std::fread(s2.data(), sizeof(char), s.length(), fp), s.length());
    ASSERT_EQ(s, s2);
    ASSERT_EQ(std::fclose(fp), 0);
  }

  std::filesystem::remove(filepath);
}
