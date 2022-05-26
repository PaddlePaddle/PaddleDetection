# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.4.1)

# Creates and names a library, sets it as either STATIC or SHARED, and provides
# the relative paths to its source code. You can define multiple libraries, and
# CMake builds them for you. Gradle automatically packages shared libraries with
# your APK.

set(PaddleLite_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../PaddleLite")
include_directories(${PaddleLite_DIR}/cxx/include)

set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../OpenCV/sdk/native/jni")
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -ffast-math -Ofast -Os -DNDEBUG -fno-exceptions -fomit-frame-pointer -fno-asynchronous-unwind-tables -fno-unwind-tables"
)
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -fvisibility=hidden -fvisibility-inlines-hidden -fdata-sections -ffunction-sections"
)
set(CMAKE_SHARED_LINKER_FLAGS
    "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--gc-sections -Wl,-z,nocopyreloc")

add_library(
  # Sets the name of the library.
  Native
  # Sets the library as a shared library.
  SHARED
  # Provides a relative path to your source file(s).
  Native.cc Pipeline.cc Utils.cc)

find_library(
  # Sets the name of the path variable.
  log-lib
  # Specifies the name of the NDK library that you want CMake to locate.
  log)

add_library(
  # Sets the name of the library.
  paddle_light_api_shared
  # Sets the library as a shared library.
  SHARED
  # Provides a relative path to your source file(s).
  IMPORTED)

set_target_properties(
  # Specifies the target library.
  paddle_light_api_shared
  # Specifies the parameter you want to define.
  PROPERTIES
    IMPORTED_LOCATION
    ${PaddleLite_DIR}/cxx/libs/${ANDROID_ABI}/libpaddle_light_api_shared.so
    # Provides the path to the library you want to import.
)

# Comment the followings if libpaddle_light_api_shared.so is not fit for NPU
add_library(
  # Sets the name of the library.
  hiai
  # Sets the library as a shared library.
  SHARED
  # Provides a relative path to your source file(s).
  IMPORTED)

set_target_properties(
  # Specifies the target library.
  hiai
  # Specifies the parameter you want to define.
  PROPERTIES IMPORTED_LOCATION
             ${PaddleLite_DIR}/cxx/libs/${ANDROID_ABI}/libhiai.so
             # Provides the path to the library you want to import.
)

add_library(
  # Sets the name of the library.
  hiai_ir
  # Sets the library as a shared library.
  SHARED
  # Provides a relative path to your source file(s).
  IMPORTED)

set_target_properties(
  # Specifies the target library.
  hiai_ir
  # Specifies the parameter you want to define.
  PROPERTIES IMPORTED_LOCATION
             ${PaddleLite_DIR}/cxx/libs/${ANDROID_ABI}/libhiai_ir.so
             # Provides the path to the library you want to import.
)

add_library(
  # Sets the name of the library.
  hiai_ir_build
  # Sets the library as a shared library.
  SHARED
  # Provides a relative path to your source file(s).
  IMPORTED)

set_target_properties(
  # Specifies the target library.
  hiai_ir_build
  # Specifies the parameter you want to define.
  PROPERTIES IMPORTED_LOCATION
             ${PaddleLite_DIR}/cxx/libs/${ANDROID_ABI}/libhiai_ir_build.so
             # Provides the path to the library you want to import.
)

# Specifies libraries CMake should link to your target library. You can link
# multiple libraries, such as libraries you define in this build script,
# prebuilt third-party libraries, or system libraries.

target_link_libraries(
  # Specifies the target library.
  Native
  paddle_light_api_shared
  ${OpenCV_LIBS}
  GLESv2
  EGL
  ${log-lib}
  hiai
  hiai_ir
  hiai_ir_build
  )

add_custom_command(
  TARGET Native
  POST_BUILD
  COMMAND
    ${CMAKE_COMMAND} -E copy
    ${PaddleLite_DIR}/cxx/libs/${ANDROID_ABI}/libc++_shared.so
    ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libc++_shared.so)

add_custom_command(
  TARGET Native
  POST_BUILD
  COMMAND
    ${CMAKE_COMMAND} -E copy
    ${PaddleLite_DIR}/cxx/libs/${ANDROID_ABI}/libpaddle_light_api_shared.so
    ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libpaddle_light_api_shared.so)

# Comment the followings if libpaddle_light_api_shared.so is not fit for NPU
add_custom_command(
  TARGET Native
  POST_BUILD
  COMMAND
    ${CMAKE_COMMAND} -E copy
    ${PaddleLite_DIR}/cxx/libs/${ANDROID_ABI}/libhiai.so
    ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libhiai.so)

add_custom_command(
  TARGET Native
  POST_BUILD
  COMMAND
    ${CMAKE_COMMAND} -E copy
    ${PaddleLite_DIR}/cxx/libs/${ANDROID_ABI}/libhiai_ir.so
    ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libhiai_ir.so)

add_custom_command(
  TARGET Native
  POST_BUILD
  COMMAND
    ${CMAKE_COMMAND} -E copy
    ${PaddleLite_DIR}/cxx/libs/${ANDROID_ABI}/libhiai_ir_build.so
    ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libhiai_ir_build.so)
