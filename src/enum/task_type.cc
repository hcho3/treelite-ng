/*!
 * Copyright (c) 2023 by Contributors
 * \file task_type.cc
 * \author Hyunsu Cho
 * \brief Utilities for TaskType enum
 */

#include <treelite/logging.h>
#include <treelite/enum/task_type.h>

#include <string>

namespace treelite {

inline std::string TaskTypeToString(TaskType type) {
  switch (type) {
  case TaskType::kBinaryClfRegr:
    return "kBinaryClfRegr";
  case TaskType::kMultiClfGrovePerClass:
    return "kMultiClfGrovePerClass";
  case TaskType::kMultiClfProbDistLeaf:
    return "kMultiClfProbDistLeaf";
  case TaskType::kMultiClfCategLeaf:
    return "kMultiClfCategLeaf";
  default:
    return "";
  }
}

inline TaskType TaskTypeFromString(std::string const& str) {
  if (str == "kBinaryClfRegr") {
    return TaskType::kBinaryClfRegr;
  } else if (str == "kMultiClfGrovePerClass") {
    return TaskType::kMultiClfGrovePerClass;
  } else if (str == "kMultiClfProbDistLeaf") {
    return TaskType::kMultiClfProbDistLeaf;
  } else if (str == "kMultiClfCategLeaf") {
    return TaskType::kMultiClfCategLeaf;
  } else {
    TREELITE_LOG(FATAL) << "Unknown task type: " << str;
    return TaskType::kBinaryClfRegr;  // to avoid compiler warning
  }
}

}  // namespace treelite
