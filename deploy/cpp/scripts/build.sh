# compile with cuda
WITH_GPU=ON
# compile with tensorrt
WITH_TENSORRT=OFF
# path to paddle inference lib
PADDLE_DIR=/root/projects/deps/fluid_inference/
# path to opencv lib
OPENCV_DIR=$(pwd)/deps/opencv346/
# path to cuda lib
CUDA_LIB=/usr/local/cuda/lib64/

sh $(pwd)/scripts/bootstrap.sh

rm -rf build
mkdir -p build
cd build
cmake .. \
    -DWITH_GPU=OFF \
    -DWITH_TENSORRT=OFF \
    -DPADDLE_DIR=${PADDLE_DIR} \
    -DCUDA_LIB=${CUDA_LIB} \
    -DOPENCV_DIR=${OPENCV_DIR}
make
