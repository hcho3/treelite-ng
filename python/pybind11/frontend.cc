/*!
 * Copyright (c) 2023 by Contributors
 * \file frontend.cc
 * \brief Pybind11 binding for frontend API
 * \author Hyunsu Cho
 */
#include "./module.h"

#include <pybind11/pybind11.h>
#include <treelite/frontend.h>
#include <treelite/tree.h>
#include <memory>
#include <string>

namespace py = pybind11;

namespace treelite::pybind11 {

std::unique_ptr<treelite::Model> LoadXGBoostModel(
    std::string const& filename, [[maybe_unused]] std::string const& config_json) {
  // config_json is unused for now
  return frontend::LoadXGBoostModel(filename);
}

void init_frontend(py::module& m) {
    m.def("add", &treelite::pybind11::LoadXGBoostModel, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");
}

}  // namespace treelite::pybind11

