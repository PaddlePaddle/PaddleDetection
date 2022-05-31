# 是否使用GPU(即是否使用 CUDA)
WITH_GPU=OFF

# 是否使用MKL or openblas，TX2需要设置为OFF
WITH_MKL=ON

# 是否集成 TensorRT(仅WITH_GPU=ON 有效)
WITH_TENSORRT=OFF

# paddle 预测库lib名称，由于不同平台不同版本预测库lib名称不同，请查看所下载的预测库中`paddle_inference/lib/`文件夹下`lib`的名称
PADDLE_LIB_NAME=libpaddle_inference

# TensorRT 的include路径
TENSORRT_INC_DIR=/path/to/tensorrt/include

# TensorRT 的lib路径
TENSORRT_LIB_DIR=/path/to/tensorrt/lib

# Paddle 预测库路径
PADDLE_DIR=/path/to/paddle_inference

# CUDA 的 lib 路径
CUDA_LIB=/path/to/cuda/lib

# CUDNN 的 lib 路径
CUDNN_LIB=/path/to/cudnn/lib


MACHINE_TYPE=`uname -m`
echo "MACHINE_TYPE: "${MACHINE_TYPE}


if [ "$MACHINE_TYPE" = "x86_64" ]
then
  echo "set OPENCV_DIR for x86_64"
  # linux系统通过以下命令下载预编译的opencv
  mkdir -p $(pwd)/deps && cd $(pwd)/deps
  wget -c https://paddledet.bj.bcebos.com/data/opencv3.4.6gcc8.2ffmpeg.zip
  unzip opencv3.4.6gcc8.2ffmpeg.zip && cd ..

  # set OPENCV_DIR
  OPENCV_DIR=$(pwd)/deps/opencv3.4.6gcc8.2ffmpeg

elif [ "$MACHINE_TYPE" = "aarch64" ]
then
  echo "set OPENCV_DIR for aarch64"
  # TX2平台通过以下命令下载预编译的opencv
  mkdir -p $(pwd)/deps && cd $(pwd)/deps
  wget -c https://paddlemodels.bj.bcebos.com/TX2_JetPack4.3_opencv_3.4.10_gcc7.5.0.zip
  unzip TX2_JetPack4.3_opencv_3.4.10_gcc7.5.0.zip && cd ..

  # set OPENCV_DIR
  OPENCV_DIR=$(pwd)/deps/TX2_JetPack4.3_opencv_3.4.10_gcc7.5.0/

else
  echo "Please set OPENCV_DIR manually"
fi

echo "OPENCV_DIR: "$OPENCV_DIR

# 以下无需改动
rm -rf build
mkdir -p build
cd build
cmake .. \
    -DWITH_GPU=${WITH_GPU} \
    -DWITH_MKL=${WITH_MKL} \
    -DWITH_TENSORRT=${WITH_TENSORRT} \
    -DTENSORRT_LIB_DIR=${TENSORRT_LIB_DIR} \
    -DTENSORRT_INC_DIR=${TENSORRT_INC_DIR} \
    -DPADDLE_DIR=${PADDLE_DIR} \
    -DWITH_STATIC_LIB=${WITH_STATIC_LIB} \
    -DCUDA_LIB=${CUDA_LIB} \
    -DCUDNN_LIB=${CUDNN_LIB} \
    -DOPENCV_DIR=${OPENCV_DIR} \
    -DPADDLE_LIB_NAME=${PADDLE_LIB_NAME}

make
echo "make finished!"
