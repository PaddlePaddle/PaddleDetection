# PicoDet MNN Demo

This fold provides PicoDet inference code using
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
- Create picodet_m_416_coco.onnx
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
   ``` shell
   python -m MNN.tools.mnnconvert -f ONNX --modelFile picodet-416.onnx --MNNModel picodet-416.mnn
   ```
Here are converted model [download link](https://paddledet.bj.bcebos.com/deploy/third_engine/picodet_m_416.mnn).

## Build

The python code *demo_mnn.py* can run directly and independently without main PicoDet repo.
`PicoDetONNX` and `PicoDetTorch` are two classes used to check the similarity of MNN inference results
with ONNX model and Pytorch model. They can be remove with no side effects.

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

## Run

### Python

`demo_mnn.py` provide an inference class `PicoDetMNN` that combines preprocess, post process, visualization.
Besides it can be used in command line with the form:

```shell
demo_mnn.py [-h] [--model_path MODEL_PATH] [--cfg_path CFG_PATH]
    [--img_fold IMG_FOLD] [--result_fold RESULT_FOLD]
    [--input_shape INPUT_SHAPE INPUT_SHAPE]
    [--backend {MNN,ONNX,torch}]
```

For example:

``` shell
# run MNN 416 model
python ./demo_mnn.py --model_path ../model/picodet-416.mnn --img_fold ../imgs --result_fold ../results
# run MNN 320 model
python ./demo_mnn.py --model_path ../model/picodet-320.mnn --input_shape 320 320 --backend MNN
# run onnx model
python ./demo_mnn.py --model_path ../model/sim.onnx --backend ONNX
```

### C++

C++ inference interface is same with NCNN code, to detect images in a fold, run:

``` shell
./picodet-mnn "1" "../imgs/test.jpg"
```

For speed benchmark

``` shell
./picodet-mnn "3" "0"
```

## Reference
[MNN](https://github.com/alibaba/MNN)
