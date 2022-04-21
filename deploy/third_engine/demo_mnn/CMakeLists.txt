cmake_minimum_required(VERSION 3.9)
project(picodet-mnn)

set(CMAKE_CXX_STANDARD 17)
set(MNN_DIR PATHS "./mnn")

# find_package(OpenCV REQUIRED PATHS "/work/dependence/opencv/opencv-3.4.3/build")
find_package(OpenCV REQUIRED)
include_directories(
        ${MNN_DIR}/include
        ${MNN_DIR}/include/MNN
        ${CMAKE_SOURCE_DIR}
)
link_directories(mnn/lib)

add_library(libMNN SHARED IMPORTED)
set_target_properties(
        libMNN
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/mnn/lib/libMNN.so
)
add_executable(picodet-mnn main.cpp picodet_mnn.cpp)
target_link_libraries(picodet-mnn MNN ${OpenCV_LIBS} libMNN.so)
