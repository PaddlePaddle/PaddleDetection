ARM_ABI = arm8#[arm7/arm8]
export ARM_ABI

ifeq ($(ARM_ABI), arm8)
    ARM_PLAT=arm64-v8a
else
    ARM_PLAT=armeabi-v7a
endif
${info ARM_ABI: ${ARM_ABI}}
${info ARM_PLAT: ${ARM_PLAT}; option[arm7/arm8]}

include ../Makefile.def

LITE_ROOT=../../../
${info LITE_ROOT: $(abspath ${LITE_ROOT})}

THIRD_PARTY_DIR=third_party
${info THIRD_PARTY_DIR: $(abspath ${THIRD_PARTY_DIR})}


OPENCV_VERSION=opencv4.1.0
OPENCV_LIBS = ${THIRD_PARTY_DIR}/${OPENCV_VERSION}/${ARM_PLAT}/libs/libopencv_imgcodecs.a \
              ${THIRD_PARTY_DIR}/${OPENCV_VERSION}/${ARM_PLAT}/libs/libopencv_imgproc.a \
              ${THIRD_PARTY_DIR}/${OPENCV_VERSION}/${ARM_PLAT}/libs/libopencv_core.a \
              ${THIRD_PARTY_DIR}/${OPENCV_VERSION}/${ARM_PLAT}/3rdparty/libs/libtegra_hal.a \
              ${THIRD_PARTY_DIR}/${OPENCV_VERSION}/${ARM_PLAT}/3rdparty/libs/liblibjpeg-turbo.a \
              ${THIRD_PARTY_DIR}/${OPENCV_VERSION}/${ARM_PLAT}/3rdparty/libs/liblibwebp.a \
              ${THIRD_PARTY_DIR}/${OPENCV_VERSION}/${ARM_PLAT}/3rdparty/libs/liblibpng.a \
              ${THIRD_PARTY_DIR}/${OPENCV_VERSION}/${ARM_PLAT}/3rdparty/libs/liblibjasper.a \
              ${THIRD_PARTY_DIR}/${OPENCV_VERSION}/${ARM_PLAT}/3rdparty/libs/liblibtiff.a \
              ${THIRD_PARTY_DIR}/${OPENCV_VERSION}/${ARM_PLAT}/3rdparty/libs/libIlmImf.a \
              ${THIRD_PARTY_DIR}/${OPENCV_VERSION}/${ARM_PLAT}/3rdparty/libs/libtbb.a \
              ${THIRD_PARTY_DIR}/${OPENCV_VERSION}/${ARM_PLAT}/3rdparty/libs/libcpufeatures.a


LITE_LIBS = -L${LITE_ROOT}/cxx/lib/ -lpaddle_light_api_shared
###############################################################
# How to use one of static libaray:                           #
#  `libpaddle_api_full_bundled.a`                             #
#  `libpaddle_api_light_bundled.a`                            #
###############################################################
# Note: default use lite's shared library.                    #
###############################################################
# 1. Comment above line using `libpaddle_light_api_shared.so`
# 2. Undo comment below line using `libpaddle_api_light_bundled.a`
# LITE_LIBS = ${LITE_ROOT}/cxx/lib/libpaddle_api_light_bundled.a

CXX_LIBS = $(LITE_LIBS) ${OPENCV_LIBS} $(SYSTEM_LIBS)

LOCAL_DIRSRCS=$(wildcard src/*.cc)
LOCAL_SRCS=$(notdir $(LOCAL_DIRSRCS))
LOCAL_OBJS=$(patsubst %.cpp, %.o, $(patsubst %.cc, %.o, $(LOCAL_SRCS)))

JSON_OBJS = json_reader.o json_value.o json_writer.o

main: $(LOCAL_OBJS) $(JSON_OBJS) fetch_opencv
	$(CC) $(SYSROOT_LINK) $(CXXFLAGS_LINK) $(LOCAL_OBJS) $(JSON_OBJS) -o main $(CXX_LIBS) $(LDFLAGS)

fetch_opencv:
	@ test -d ${THIRD_PARTY_DIR} ||  mkdir ${THIRD_PARTY_DIR}
	@ test -e ${THIRD_PARTY_DIR}/${OPENCV_VERSION}.tar.gz || \
      (echo "fetch opencv libs" && \
      wget -P ${THIRD_PARTY_DIR} https://paddle-inference-dist.bj.bcebos.com/${OPENCV_VERSION}.tar.gz)
	@ test -d ${THIRD_PARTY_DIR}/${OPENCV_VERSION} || \
      tar -zxf ${THIRD_PARTY_DIR}/${OPENCV_VERSION}.tar.gz -C ${THIRD_PARTY_DIR}

fetch_json_code:
	@ test -d ${THIRD_PARTY_DIR} ||  mkdir ${THIRD_PARTY_DIR}
	@ test -e ${THIRD_PARTY_DIR}/jsoncpp_code.tar.gz || \
      (echo "fetch jsoncpp_code.tar.gz" && \
      wget -P ${THIRD_PARTY_DIR} https://bj.bcebos.com/v1/paddledet/deploy/jsoncpp_code.tar.gz )
	@ test -d ${THIRD_PARTY_DIR}/jsoncpp_code || \
      tar -zxf ${THIRD_PARTY_DIR}/jsoncpp_code.tar.gz -C ${THIRD_PARTY_DIR}

LOCAL_INCLUDES = -I./ -Iinclude
OPENCV_INCLUDE = -I${THIRD_PARTY_DIR}/${OPENCV_VERSION}/${ARM_PLAT}/include
JSON_INCLUDE = -I${THIRD_PARTY_DIR}/jsoncpp_code/include
CXX_INCLUDES = ${LOCAL_INCLUDES} ${INCLUDES} ${OPENCV_INCLUDE} ${JSON_INCLUDE} -I$(LITE_ROOT)/cxx/include


$(LOCAL_OBJS): %.o: src/%.cc fetch_opencv fetch_json_code
	$(CC) $(SYSROOT_COMPLILE) $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -c $< -o $@

$(JSON_OBJS): %.o: ${THIRD_PARTY_DIR}/jsoncpp_code/%.cpp fetch_json_code
	$(CC) $(SYSROOT_COMPLILE) $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -c $< -o $@

.PHONY: clean fetch_opencv fetch_json_code
clean:
	rm -rf $(LOCAL_OBJS) $(JSON_OBJS)
	rm -f main
