cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 11)
set(PYBIND11_CPP_STANDARD -std=c++11)

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY "https://github.com/pybind/pybind11.git")
FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
  FetchContent_Populate(pybind11)
  add_subdirectory(${pybind11_SOURCE_DIR})
endif()

set(nms_SOURCES src/binding.cc src/nms.cc)
add_library(nms MODULE ${nms_SOURCES})
set_target_properties(nms PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
    OUTPUT_NAME "nms"
)
target_compile_options(nms BEFORE PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-g -O2 -fopenmp -mavx2>)
target_link_libraries(nms PRIVATE -fopenmp)
target_link_libraries(nms PRIVATE pybind11::module)

execute_process(COMMAND ${PYTHON_EXECUTABLE} "-c" "import sysconfig; print(sysconfig.get_config_vars()['INCLUDEPY'])" OUTPUT_VARIABLE PYTHON_INC_PATHS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${PYTHON_EXECUTABLE} "-c" "import sysconfig; print(sysconfig.get_config_vars()['BLDLIBRARY'])" OUTPUT_VARIABLE PYTHON_LIB_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${PYTHON_EXECUTABLE} "-c" "import sysconfig; print(sysconfig.get_config_vars().get('EXT_SUFFIX', 'so'))" OUTPUT_VARIABLE EXT_SUFFIX OUTPUT_STRIP_TRAILING_WHITESPACE)

install(TARGETS nms DESTINATION ${CMAKE_HOME_DIRECTORY}/ppdet/modeling)
