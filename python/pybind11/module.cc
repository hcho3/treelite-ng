/*!
 * Copyright (c) 2023 by Contributors
 * \file module.cc
 * \brief Define Pybind11 module
 * \author Hyunsu Cho
 */

#include "./module.h"

#include <pybind11/pybind11.h>
#include <treelite/error.h>
#include <treelite/version.h>

namespace py = pybind11;

PYBIND11_MODULE(_ext, m) {
  m.attr("__version__") = TREELITE_VERSION;
  treelite::pybind11::init_frontend(m);
  treelite::pybind11::init_tree(m);
  py::register_local_exception<treelite::Error>(m, "TreeliteError", PyExc_RuntimeError);
}
