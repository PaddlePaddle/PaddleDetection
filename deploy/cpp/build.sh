rm -rf build
mkdir -p build
cd build
make clean
cmake .. \
    -DWITH_GPU=ON \
    -DWITH_MKL=ON \
    -DWITH_TENSORRT=OFF \
    -DPADDLE_DIR=/root/projects/deps/fluid_inference/ \
    -DCUDA_LIB=/usr/local/cuda/lib64/ \
    -DCUDNN_LIB=/usr/local/cuda/lib64/ \
    -DOPENCV_DIR=/root/projects/deps/opencv346/
make
