cmake_minimum_required(VERSION 3.9)
set(CMAKE_CXX_STANDARD 17)

project(picodet_demo)

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# find_package(OpenCV REQUIRED)
find_package(OpenCV REQUIRED PATHS "/path/to/opencv-3.4.16_gcc8.2_ffmpeg")

# find_package(ncnn REQUIRED)
find_package(ncnn REQUIRED PATHS "/path/to/ncnn/build/install/lib/cmake/ncnn")
if(NOT TARGET ncnn)
    message(WARNING "ncnn NOT FOUND!  Please set ncnn_DIR environment variable")
else()
    message("ncnn FOUND ")
endif()

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)


add_executable(picodet_demo main.cpp picodet.cpp)

target_link_libraries(
    picodet_demo
    ncnn
    ${OpenCV_LIBS}
)
