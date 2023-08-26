/*!
 * Copyright (c) 2023 by Contributors
 * \file tree.cc
 * \brief Pybind11 binding for Tree object
 * \author Hyunsu Cho
 */
#include "./module.h"

#include <pybind11/pybind11.h>
#include <treelite/tree.h>
#include <string>

namespace py = pybind11;

namespace treelite::pybind11 {

std::string DumpAsJSON(treelite::Model& model) {
    return model.DumpAsJSON(true);
}

void init_tree(py::module& m) {
    m.def("add", &DumpAsJSON, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");
}

}  // namespace treelite::pybind11