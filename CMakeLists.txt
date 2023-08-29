cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(treelite VERSION 0.0.1 LANGUAGES CXX)

if(DEFINED ENV{CONDA_PREFIX})
  set(CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX};${CMAKE_PREFIX_PATH}")
  message(STATUS "Detected Conda environment, CMAKE_PREFIX_PATH set to: ${CMAKE_PREFIX_PATH}")
  if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT OR USING_CONDA_INSTALL_PREFIX)
    message(STATUS "No CMAKE_INSTALL_PREFIX argument detected, setting to: $ENV{CONDA_PREFIX}")
    set(CMAKE_INSTALL_PREFIX $ENV{CONDA_PREFIX})
    set(USING_CONDA_INSTALL_PREFIX ON CACHE BOOL "Installing into Conda prefix")
  endif()
else()
  message(STATUS "No Conda environment detected")
endif()

include(cmake/ExternalLibs.cmake)
include(cmake/Version.cmake)
write_version()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

add_library(objtreelite OBJECT)
target_sources(objtreelite
    PRIVATE
    include/treelite/contiguous_array.h
    include/treelite/error.h
    include/treelite/frontend.h
    include/treelite/logging.h
    include/treelite/pybuffer_frame.h
    include/treelite/thread_local.h
    include/treelite/tree.h
    include/treelite/detail/contiguous_array.h
    include/treelite/detail/tree.h
    include/treelite/enum/operator.h
    include/treelite/enum/task_type.h
    include/treelite/enum/tree_node_type.h
    include/treelite/enum/typeinfo.h
    src/json_serializer.cc
    src/logging.cc
    src/enum/operator.cc
    src/enum/task_type.cc
    src/enum/tree_node_type.cc
    src/enum/typeinfo.cc
    src/frontend/xgboost.cc
    src/frontend/detail/common.h
    src/frontend/detail/xgboost.cc
    src/frontend/detail/xgboost.h
)
set_target_properties(objtreelite
    PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
target_link_libraries(objtreelite PRIVATE RapidJSON::rapidjson)
target_compile_features(objtreelite PUBLIC cxx_std_17)
target_include_directories(objtreelite PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>)

add_library(treelite SHARED)
target_link_libraries(treelite PRIVATE objtreelite)

python_add_library(_ext MODULE WITH_SOABI)
target_sources(_ext
    PRIVATE
    python/pybind11/frontend.cc
    python/pybind11/logging.cc
    python/pybind11/module.cc
    python/pybind11/module.h
    python/pybind11/tree.cc
)
target_link_libraries(_ext PRIVATE pybind11::headers objtreelite)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
if(SKBUILD)
  install(
      TARGETS _ext
      DESTINATION treelite
  )
else()
  install(
      TARGETS treelite objtreelite
      EXPORT TreeliteTargets
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
      INCLUDES DESTINATION include
  )
  install(DIRECTORY include/treelite ${PROJECT_BINARY_DIR}/include/treelite
      DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()
