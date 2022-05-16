# PicoDet NCNN Demo

该Demo提供的预测代码是根据[Tencent's NCNN framework](https://github.com/Tencent/ncnn)推理库预测的。

# 第一步：编译
## Windows
### Step1.
Download and Install Visual Studio from https://visualstudio.microsoft.com/vs/community/

### Step2.
Download and install OpenCV from https://github.com/opencv/opencv/releases

为了方便，如果环境是gcc8.2 x86环境，可直接下载以下库：
```shell
wget https://paddledet.bj.bcebos.com/data/opencv-3.4.16_gcc8.2_ffmpeg.tar.gz
tar -xf opencv-3.4.16_gcc8.2_ffmpeg.tar.gz
```

### Step3(可选).
Download and install Vulkan SDK from https://vulkan.lunarg.com/sdk/home

### Step4：编译NCNN

``` shell script
git clone --recursive https://github.com/Tencent/ncnn.git
```
Build NCNN following this tutorial: [Build for Windows x64 using VS2017](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-windows-x64-using-visual-studio-community-2017)

### Step5.

增加 `ncnn_DIR` = `YOUR_NCNN_PATH/build/install/lib/cmake/ncnn` 到系统变量中

Build project: Open x64 Native Tools Command Prompt for VS 2019 or 2017

``` cmd
cd <this-folder>
mkdir -p build
cd build
cmake ..
msbuild picodet_demo.vcxproj /p:configuration=release /p:platform=x64
```

## Linux

### Step1.
Build and install OpenCV from https://github.com/opencv/opencv

### Step2(可选).
Download Vulkan SDK from https://vulkan.lunarg.com/sdk/home

### Step3：编译NCNN
Clone NCNN repository

``` shell script
git clone --recursive https://github.com/Tencent/ncnn.git
```

Build NCNN following this tutorial: [Build for Linux / NVIDIA Jetson / Raspberry Pi](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-linux)

### Step4：编译可执行文件

``` shell script
cd <this-folder>
mkdir build
cd build
cmake ..
make
```
# Run demo

- 准备模型
    ```shell
    modelName=picodet_s_320_coco_lcnet
    # 导出Inference model
    python tools/export_model.py \
            -c configs/picodet/${modelName}.yml \
            -o weights=${modelName}.pdparams \
            --output_dir=inference_model
    # 转换到ONNX
    paddle2onnx --model_dir inference_model/${modelName} \
            --model_filename model.pdmodel  \
            --params_filename model.pdiparams \
            --opset_version 11 \
            --save_file ${modelName}.onnx
    # 简化模型
    python -m onnxsim ${modelName}.onnx ${modelName}_processed.onnx
    # 将模型转换至NCNN格式
    Run onnx2ncnn in ncnn tools to generate ncnn .param and .bin file.
    ```
转NCNN模型可以利用在线转换工具 [https://convertmodel.com](https://convertmodel.com/)

为了快速测试，可直接下载：[picodet_s_320_coco_lcnet-opt.bin](https://paddledet.bj.bcebos.com/deploy/third_engine/picodet_s_320_coco_lcnet-opt.bin)/ [picodet_s_320_coco_lcnet-opt.param](https://paddledet.bj.bcebos.com/deploy/third_engine/picodet_s_320_coco_lcnet-opt.param)（不带后处理）。

**注意：**由于带后处理后，NCNN预测会出NAN，暂时使用不带后处理Demo即可，带后处理的Demo正在升级中，很快发布。


## 开始运行

首先新建预测结果存放目录：
```shell
cp -r ../demo_onnxruntime/imgs .
cd build
mkdir ../results
```

- 预测一张图片
``` shell
./picodet_demo 0 ../picodet_s_320_coco_lcnet.bin ../picodet_s_320_coco_lcnet.param 320 320 ../imgs/dog.jpg 0
```
具体参数解析可参考`main.cpp`。

-测试速度Benchmark

``` shell
./picodet_demo 1 ../picodet_s_320_lcnet.bin ../picodet_s_320_lcnet.param 320 320  0
```

## FAQ

- 预测结果精度不对：
请先确认模型输入shape是否对齐，并且模型输出name是否对齐，不带后处理的PicoDet增强版模型输出name如下：
```shell
# 分类分支  |  检测分支
{"transpose_0.tmp_0", "transpose_1.tmp_0"},
{"transpose_2.tmp_0", "transpose_3.tmp_0"},
{"transpose_4.tmp_0", "transpose_5.tmp_0"},
{"transpose_6.tmp_0", "transpose_7.tmp_0"},
```
可使用[netron](https://netron.app)查看具体name，并修改`picodet_mnn.hpp`中相应`non_postprocess_heads_info`数组。
