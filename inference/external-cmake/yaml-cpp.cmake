
find_package(Git REQUIRED)

include(ExternalProject)

message("${CMAKE_BUILD_TYPE}")

ExternalProject_Add(
        ext-yaml-cpp
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG e0e01d53c27ffee6c86153fa41e7f5e57d3e5c90
        CMAKE_ARGS
        -DYAML_CPP_BUILD_TESTS=OFF
		-DYAML_CPP_BUILD_TOOLS=OFF
        -DYAML_CPP_INSTALL=OFF
        -DYAML_CPP_BUILD_CONTRIB=OFF
		-DMSVC_SHARED_RT=OFF
		-DBUILD_SHARED_LIBS=OFF
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
        -DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}
        -DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}
        -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=${CMAKE_BINARY_DIR}/ext/yaml-cpp/lib
        -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=${CMAKE_BINARY_DIR}/ext/yaml-cpp/lib
        PREFIX "${CMAKE_BINARY_DIR}/ext/yaml-cpp"
        # Disable install step
        INSTALL_COMMAND ""
	    LOG_DOWNLOAD ON
)
