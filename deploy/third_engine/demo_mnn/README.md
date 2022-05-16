# PicoDet MNN Demo

本Demo提供的预测代码是根据[Alibaba's MNN framework](https://github.com/alibaba/MNN) 推理库预测的。

## C++ Demo

- 第一步：根据[MNN官方编译文档](https://www.yuque.com/mnn/en/build_linux) 编译生成预测库.
- 第二步：编译或下载得到OpenCV库，可参考OpenCV官网，为了方便如果环境是gcc8.2 x86环境，可直接下载以下库：
```shell
wget https://paddledet.bj.bcebos.com/data/opencv-3.4.16_gcc8.2_ffmpeg.tar.gz
tar -xf opencv-3.4.16_gcc8.2_ffmpeg.tar.gz
```

- 第三步：准备模型
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
    # 将模型转换至MNN格式
    python -m MNN.tools.mnnconvert -f ONNX --modelFile picodet_s_320_lcnet_processed.onnx --MNNModel picodet_s_320_lcnet.mnn
    ```
为了快速测试，可直接下载：[picodet_s_320_lcnet.mnn](https://paddledet.bj.bcebos.com/deploy/third_engine/picodet_s_320_lcnet.mnn)（不带后处理）。

**注意：**由于MNN里，Matmul算子的输入shape如果不一致计算有问题，带后处理的Demo正在升级中，很快发布。

## 编译可执行程序

- 第一步：导入lib包
```
mkdir mnn && cd mnn && mkdir lib
cp /path/to/MNN/build/libMNN.so .
cd ..
cp -r /path/to/MNN/include .
```
- 第二步：修改CMakeLists.txt中OpenCV和MNN的路径
- 第三步：开始编译
``` shell
mkdir build && cd build
cmake ..
make
```
如果在build目录下生成`picodet-mnn`可执行文件，就证明成功了。

## 开始运行

首先新建预测结果存放目录：
```shell
cp -r ../demo_onnxruntime/imgs .
cd build
mkdir ../results
```

- 预测一张图片
``` shell
./picodet-mnn 0 ../picodet_s_320_lcnet_3.mnn 320 320 ../imgs/dog.jpg
```

-测试速度Benchmark

``` shell
./picodet-mnn 1 ../picodet_s_320_lcnet.mnn 320 320
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

## Reference
[MNN](https://github.com/alibaba/MNN)
