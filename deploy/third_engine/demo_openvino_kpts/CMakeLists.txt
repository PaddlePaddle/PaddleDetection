cmake_minimum_required(VERSION 3.4.1)
set(CMAKE_CXX_STANDARD 14)

project(tinypose_demo)

find_package(OpenCV REQUIRED)
find_package(InferenceEngine REQUIRED)
find_package(ngraph REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)

add_executable(tinypose_demo main.cpp picodet_openvino.cpp keypoint_detector.cpp keypoint_postprocess.cpp)

target_link_libraries(
    tinypose_demo
    ${InferenceEngine_LIBRARIES}
    ${NGRAPH_LIBRARIES}
    ${OpenCV_LIBS}
)
