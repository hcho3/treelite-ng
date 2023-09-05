# Object library

add_library(objtreelite OBJECT)
target_link_libraries(objtreelite PRIVATE RapidJSON::rapidjson)

if(USE_OPENMP)
  if(APPLE)
    find_package(OpenMP)
    if (NOT OpenMP_FOUND)
      # Try again with extra path info; required for libomp 15+ from Homebrew
      message(STATUS "OpenMP not found; attempting to locate libomp from Homebrew...")
      execute_process(COMMAND brew --prefix libomp
          OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
          OUTPUT_STRIP_TRAILING_WHITESPACE)
      set(OpenMP_C_FLAGS
          "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
      set(OpenMP_CXX_FLAGS
          "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
      set(OpenMP_C_LIB_NAMES omp)
      set(OpenMP_CXX_LIB_NAMES omp)
      set(OpenMP_omp_LIBRARY ${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib)
      find_package(OpenMP REQUIRED)
    endif()
  else()
    find_package(OpenMP REQUIRED)
  endif()
else()
  message(STATUS "Disabling OpenMP")
endif()

target_include_directories(objtreelite PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>)
if(MSVC)
  target_compile_options(objtreelite PRIVATE /MP)
  target_compile_definitions(objtreelite PRIVATE -DNOMINMAX)
  target_compile_options(objtreelite PRIVATE /utf-8 -D_CRT_SECURE_NO_WARNINGS -D_CRT_SECURE_NO_DEPRECATE)
else()
  target_compile_options(objtreelite PRIVATE -funroll-loops)
endif()
if(USE_OPENMP)
  target_link_libraries(objtreelite PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(objtreelite PUBLIC -DTREELITE_OPENMP_SUPPORT)
endif()

set_target_properties(objtreelite
    PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
target_compile_features(objtreelite PUBLIC cxx_std_17)

target_sources(objtreelite
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include/treelite/c_api.h
    ${PROJECT_SOURCE_DIR}/include/treelite/c_api_error.h
    ${PROJECT_SOURCE_DIR}/include/treelite/contiguous_array.h
    ${PROJECT_SOURCE_DIR}/include/treelite/error.h
    ${PROJECT_SOURCE_DIR}/include/treelite/frontend.h
    ${PROJECT_SOURCE_DIR}/include/treelite/gtil.h
    ${PROJECT_SOURCE_DIR}/include/treelite/logging.h
    ${PROJECT_SOURCE_DIR}/include/treelite/pybuffer_frame.h
    ${PROJECT_SOURCE_DIR}/include/treelite/thread_local.h
    ${PROJECT_SOURCE_DIR}/include/treelite/tree.h
    ${PROJECT_SOURCE_DIR}/include/treelite/detail/contiguous_array.h
    ${PROJECT_SOURCE_DIR}/include/treelite/detail/tree.h
    ${PROJECT_SOURCE_DIR}/include/treelite/enum/operator.h
    ${PROJECT_SOURCE_DIR}/include/treelite/enum/task_type.h
    ${PROJECT_SOURCE_DIR}/include/treelite/enum/tree_node_type.h
    ${PROJECT_SOURCE_DIR}/include/treelite/enum/typeinfo.h
    json_serializer.cc
    logging.cc
    c_api/c_api_utils.h
    c_api/c_api.cc
    c_api/c_api_error.cc
    c_api/frontend.cc
    c_api/logging.cc
    c_api/model.cc
    enum/operator.cc
    enum/task_type.cc
    enum/tree_node_type.cc
    enum/typeinfo.cc
    frontend/xgboost_json.cc
    frontend/xgboost_legacy.cc
    frontend/detail/common.h
    frontend/detail/xgboost.cc
    frontend/detail/xgboost.h
    frontend/detail/xgboost_json.h
    gtil/config.cc
    gtil/predict.cc
    gtil/detail/omp_exception.h
    gtil/detail/threading_utils.h
)
