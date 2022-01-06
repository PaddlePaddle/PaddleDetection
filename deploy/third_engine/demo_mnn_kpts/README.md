# TinyPose MNN Demo

This fold provides PicoDet+TinyPose inference code using
[Alibaba's MNN framework](https://github.com/alibaba/MNN). Most of the implements in
this fold are same as *demo_ncnn*.

## Install MNN

### Python library

Just run:

``` shell
pip install MNN
```

### C++ library

Please follow the [official document](https://www.yuque.com/mnn/en/build_linux) to build MNN engine.

- Create picodet_m_416_coco.onnx and tinypose256.onnx
    example:
    ```shell
    modelName=picodet_m_416_coco
    # export model
    python tools/export_model.py \
            -c configs/picodet/${modelName}.yml \
            -o weights=${modelName}.pdparams \
            --output_dir=inference_model
    # convert to onnx
    paddle2onnx --model_dir inference_model/${modelName} \
            --model_filename model.pdmodel  \
            --params_filename model.pdiparams \
            --opset_version 11 \
            --save_file ${modelName}.onnx
    # onnxsim
    python -m onnxsim ${modelName}.onnx ${modelName}_processed.onnx
    ```

- Convert model
    example:
    ``` shell
    python -m MNN.tools.mnnconvert -f ONNX --modelFile picodet-416.onnx --MNNModel picodet-416.mnn
    ```
Here are converted model
[picodet_m_416](https://paddledet.bj.bcebos.com/deploy/third_engine/picodet_m_416.mnn).
[tinypose256](https://paddledet.bj.bcebos.com/deploy/third_engine/tinypose256.mnn)

## Build

For C++ code, replace `libMNN.so` under *./mnn/lib* with the one you just compiled, modify OpenCV path and MNN path at CMake file,
and run

``` shell
mkdir build && cd build
cmake ..
make
```

Note that a flag at `main.cpp` is used to control whether to show the detection result or save it into a fold.

``` c++
#define __SAVE_RESULT__ // if defined save drawed results to ../results, else show it in windows
```

#### ARM Build

Prepare OpenCV library [OpenCV_4_1](https://paddle-inference-dist.bj.bcebos.com/opencv4.1.0.tar.gz).

``` shell
mkdir third && cd third
wget https://paddle-inference-dist.bj.bcebos.com/opencv4.1.0.tar.gz
tar -zxvf opencv4.1.0.tar.gz
cd ..

mkdir build && cd build
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI="arm64-v8a" -DANDROID_PLATFORM=android-21 -DANDROID_TOOLCHAIN=gcc ..
make
```

## Run

To detect images in a fold, run:
``` shell
./tinypose-mnn [mode] [image_file]
```
|  param   | detail  |
|  ----  | ----  |
| --mode  | input mode，0:camera；1:image；2:video；3:benchmark |
| --image_file  | input image path |

for example:

``` shell
./tinypose-mnn "1" "../imgs/test.jpg"
```

For speed benchmark:

``` shell
./tinypose-mnn "3" "0"
```

## Benchmark
Plateform: Kirin980
Model: [tinypose256](https://paddledet.bj.bcebos.com/deploy/third_engine/tinypose256.mnn)

| param    | Min(s) | Max(s) | Avg(s) |
| -------- | ------ | ------ | ------ |
| Thread=4 | 0.018  | 0.021  | 0.019  |
| Thread=1 | 0.031  | 0.041  | 0.032  |



## Reference
[MNN](https://github.com/alibaba/MNN)
