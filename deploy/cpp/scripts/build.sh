# 是否使用GPU(即是否使用 CUDA)
WITH_GPU=OFF

# 是否使用MKL or openblas
WITH_MKL=ON

# 是否集成 TensorRT(仅WITH_GPU=ON 有效)
WITH_TENSORRT=OFF

# TensorRT 的路径
TENSORRT_DIR=/path/to/TensorRT/

# Paddle 预测库路径
PADDLE_DIR=/path/to/fluid_inference/

# Paddle 的预测库是否使用静态库来编译
# 使用TensorRT时，Paddle的预测库通常为动态库
WITH_STATIC_LIB=OFF

# CUDA 的 lib 路径
CUDA_LIB=/path/to/cuda/lib/

# CUDNN 的 lib 路径
CUDNN_LIB=/path/to/cudnn/lib/

# OPENCV 路径, 如果使用自带预编译版本可不修改
if [ "$(uname -m)"=="x86_64" ]; then
  # linux系统通过以下命令下载预编译的opencv
  mkdir -p $(pwd)/deps && cd $(pwd)/deps
  wget -c https://bj.bcebos.com/paddleseg/deploy/opencv3.4.6gcc4.8ffmpeg.tar.gz2
  tar xvfj opencv3.4.6gcc4.8ffmpeg.tar.gz2 && cd ..

  # set OPENCV_DIR
  OPENCV_DIR=$(pwd)/deps/opencv3.4.6gcc4.8ffmpeg/

elif [ "$(uname -m)"=="aarch64" ];then
  # TX2平台通过以下命令下载预编译的opencv
  mkdir -p $(pwd)/deps && cd $(pwd)/deps
  wget -c https://paddlemodels.bj.bcebos.com/TX2_JetPack4.3_opencv_3.4.10_gcc7.5.0.zip
  unzip TX2_JetPack4.3_opencv_3.4.10_gcc7.5.0.zip && cd ..

  # set OPENCV_DIR
  OPENCV_DIR=$(pwd)/deps/TX2_JetPack4.3_opencv_3.4.10_gcc7.5.0/

fi

# 请检查以上各个路径是否正确

# 以下无需改动
rm -rf build
mkdir -p build
cd build
cmake .. \
    -DWITH_GPU=${WITH_GPU} \
    -DWITH_MKL=${WITH_MKL} \
    -DWITH_TENSORRT=${WITH_TENSORRT} \
    -DTENSORRT_DIR=${TENSORRT_DIR} \
    -DPADDLE_DIR=${PADDLE_DIR} \
    -DWITH_STATIC_LIB=${WITH_STATIC_LIB} \
    -DCUDA_LIB=${CUDA_LIB} \
    -DCUDNN_LIB=${CUDNN_LIB} \
    -DOPENCV_DIR=${OPENCV_DIR}

make
echo "make finished!"
