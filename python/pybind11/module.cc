/*!
 * Copyright (c) 2023 by Contributors
 * \file module.cc
 * \brief Define Pybind11 module
 * \author Hyunsu Cho
 */

#include "./module.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_ext, m) {
    m.doc() = R"pbdoc(
        Treelite
        --------

        .. currentmodule:: treelite

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

    treelite::pybind11::init_frontend(m);
    treelite::pybind11::init_tree(m);
}