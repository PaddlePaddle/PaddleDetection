cmake_minimum_required(VERSION 3.9)

project(tinypose-mnn)

set(CMAKE_CXX_STANDARD 17)
set(MNN_DIR {YOUR_MNN_DIR})

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
add_executable(tinypose-mnn main.cpp picodet_mnn.cpp keypoint_detector.cpp keypoint_postprocess.cpp)

target_link_libraries(tinypose-mnn MNN ${OpenCV_LIBS} libMNN.so)

