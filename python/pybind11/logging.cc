/*!
 * Copyright (c) 2023 by Contributors
 * \file logging.cc
 * \brief Pybind11 binding for logger
 * \author Hyunsu Cho
 */
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <treelite/logging.h>

#include <functional>

#include "./module.h"

namespace py = pybind11;

namespace treelite::pybind11 {

using Callback = std::function<void(std::string const&)>;

void init_logging(py::module& m) {
  m.def("register_callback_log_info", [](Callback const& callback) {
    treelite::LogCallbackRegistryStore::Get()->RegisterCallBackLogInfo(callback);
  });
  m.def("register_callback_log_warning", [](Callback const& callback) {
    treelite::LogCallbackRegistryStore::Get()->RegisterCallBackLogWarning(callback);
  });
}

}  // namespace treelite::pybind11
