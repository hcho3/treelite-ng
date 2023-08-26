/*!
 * Copyright (c) 2023 by Contributors
 * \file module.h
 * \brief Defines Pybind11 module
 * \author Hyunsu Cho
 */

#ifndef PYTHON_PYBIND11_MODULE_H_
#define PYTHON_PYBIND11_MODULE_H_

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace treelite::pybind11 {
void init_frontend(py::module& m);
void init_tree(py::module& m);
}  // namespace treelite::pybind11

#endif  // PYTHON_PYBIND11_MODULE_H_
