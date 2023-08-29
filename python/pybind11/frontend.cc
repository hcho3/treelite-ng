/*!
 * Copyright (c) 2023 by Contributors
 * \file frontend.cc
 * \brief Pybind11 binding for frontend API
 * \author Hyunsu Cho
 */
#include <pybind11/pybind11.h>
#include <treelite/frontend.h>
#include <treelite/tree.h>

#include <memory>
#include <string>

#include "./module.h"

namespace py = pybind11;

namespace treelite::pybind11 {

std::unique_ptr<treelite::Model> LoadXGBoostModel(
    std::string const& filename, [[maybe_unused]] std::string const& config_json) {
  // config_json is unused for now
  return frontend::LoadXGBoostModel(filename);
}

void init_frontend(py::module& m) {
  m.def("load_xgboost_model", &treelite::pybind11::LoadXGBoostModel, "");
}

}  // namespace treelite::pybind11
