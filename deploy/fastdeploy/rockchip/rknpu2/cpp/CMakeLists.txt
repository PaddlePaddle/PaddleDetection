CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
project(infer_demo)

set(CMAKE_CXX_STANDARD 14)

option(FASTDEPLOY_INSTALL_DIR "Path of downloaded fastdeploy sdk.")

include(${FASTDEPLOY_INSTALL_DIR}/FastDeployConfig.cmake)
include_directories(${FastDeploy_INCLUDE_DIRS})
add_executable(infer_demo infer.cc)
target_link_libraries(infer_demo ${FastDeploy_LIBS})
