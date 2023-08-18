#include <treelite/tree.h>
#include <treelite/frontend.h>
#include <pybind11/pybind11.h>

#include <memory>
#include <string>

namespace py = pybind11;

namespace treelite::pybind11 {

std::unique_ptr<treelite::Model> LoadXGBoostModel(
    const std::string& filename, [[maybe_unused]] const std::string& config_json
) {
  // config_json is unused for now
  return frontend::LoadXGBoostModel(filename);
}

}  // namespace treelite::pybind11

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

m.def("add", &treelite::pybind11::LoadXGBoostModel, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");
}
