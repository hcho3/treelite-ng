/*!
 * Copyright (c) 2023 by Contributors
 * \file tree.cc
 * \brief Pybind11 binding for Tree object
 * \author Hyunsu Cho
 */
#include <pybind11/pybind11.h>
#include <treelite/tree.h>

#include <string>

#include "./module.h"

namespace py = pybind11;

namespace treelite::pybind11 {

std::string DumpAsJSON(treelite::Model& model, bool pretty_print) {
  return model.DumpAsJSON(pretty_print);
}

void init_tree(py::module& m) {
  m.def("dump_as_json", &DumpAsJSON, "");
}

}  // namespace treelite::pybind11
