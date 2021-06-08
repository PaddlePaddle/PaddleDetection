# Jetson平台编译指南

## 说明
`NVIDIA Jetson`设备是具有`NVIDIA GPU`的嵌入式设备，可以将目标检测算法部署到该设备上。本文档是在`Jetson`硬件上部署`PaddleDetection`模型的教程。

本文档以`Jetson TX2`硬件、`JetPack 4.3`版本为例进行说明。

`Jetson`平台的开发指南请参考[NVIDIA Jetson Linux Developer Guide](https://docs.nvidia.com/jetson/l4t/index.html).

## Jetson环境搭建
`Jetson`系统软件安装，请参考[NVIDIA Jetson Linux Developer Guide](https://docs.nvidia.com/jetson/l4t/index.html).

* (1) 查看硬件系统的l4t的版本号
```
cat /etc/nv_tegra_release
```
* (2) 根据硬件，选择硬件可安装的`JetPack`版本，硬件和`JetPack`版本对应关系请参考[jetpack-archive](https://developer.nvidia.com/embedded/jetpack-archive).

* (3) 下载`JetPack`，请参考[NVIDIA Jetson Linux Developer Guide](https://docs.nvidia.com/jetson/l4t/index.html) 中的`Preparing a Jetson Developer Kit for Use`章节内容进行刷写系统镜像。

## 下载或编译`Paddle`预测库
本文档使用`Paddle`在`JetPack4.3`上预先编译好的预测库，请根据硬件在[安装与编译 Linux 预测库](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/05_inference_deployment/inference/build_and_install_lib_cn.html) 中选择对应版本的`Paddle`预测库。

这里选择[nv_jetson_cuda10_cudnn7.6_trt6(jetpack4.3)](https://paddle-inference-lib.bj.bcebos.com/2.0.0-nv-jetson-jetpack4.3-all/paddle_inference.tgz), `Paddle`版本`2.0.0-rc0`,`CUDA`版本`10.0`,`CUDNN`版本`7.6`，`TensorRT`版本`6`。

若需要自己在`Jetson`平台上自定义编译`Paddle`库，请参考文档[安装与编译 Linux 预测库](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/inference/build_and_install_lib_cn.html) 的`NVIDIA Jetson嵌入式硬件预测库源码编译`部分内容。

### Step1: 下载代码

 `git clone https://github.com/PaddlePaddle/PaddleDetection.git`

**说明**：其中`C++`预测代码在`/root/projects/PaddleDetection/deploy/cpp` 目录，该目录不依赖任何`PaddleDetection`下其他目录。


### Step2: 下载PaddlePaddle C++ 预测库 fluid_inference

解压下载的[nv_jetson_cuda10_cudnn7.6_trt6(jetpack4.3)](https://paddle-inference-lib.bj.bcebos.com/2.0.1-nv-jetson-jetpack4.3-all/paddle_inference.tgz) 。

下载并解压后`/root/projects/fluid_inference`目录包含内容为：
```
fluid_inference
├── paddle # paddle核心库和头文件
|
├── third_party # 第三方依赖库和头文件
|
└── version.txt # 版本和编译信息
```

**注意:** 预编译库`nv-jetson-cuda10-cudnn7.6-trt6`使用的`GCC`版本是`7.5.0`，其他都是使用`GCC 4.8.5`编译的。使用高版本的GCC可能存在`ABI`兼容性问题，建议降级或[自行编译预测库](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/inference/build_and_install_lib_cn.html)。


### Step4: 编译

编译`cmake`的命令在`scripts/build.sh`中，请根据实际情况修改主要参数，其主要内容说明如下：

注意，`TX2`平台的`CUDA`、`CUDNN`需要通过`JetPack`安装。

```
# 是否使用GPU(即是否使用 CUDA)
WITH_GPU=ON

# 是否使用MKL or openblas，TX2需要设置为OFF
WITH_MKL=OFF

# 是否集成 TensorRT(仅WITH_GPU=ON 有效)
WITH_TENSORRT=ON

# TensorRT 的include路径
TENSORRT_INC_DIR=/usr/include/aarch64-linux-gnu

# TensorRT 的lib路径
TENSORRT_LIB_DIR=/usr/lib/aarch64-linux-gnu

# Paddle 预测库路径
PADDLE_DIR=/path/to/fluid_inference/

# Paddle 预测库名称
PADDLE_LIB_NAME=paddle_inference

# Paddle 的预测库是否使用静态库来编译
# 使用TensorRT时，Paddle的预测库通常为动态库
WITH_STATIC_LIB=OFF

# CUDA 的 lib 路径
CUDA_LIB=/usr/local/cuda-10.0/lib64

# CUDNN 的 lib 路径
CUDNN_LIB=/usr/lib/aarch64-linux-gnu

# OPENCV_DIR 的路径
# linux平台请下载：https://bj.bcebos.com/paddleseg/deploy/opencv3.4.6gcc4.8ffmpeg.tar.gz2，并解压到deps文件夹下
# TX2平台请下载：https://paddlemodels.bj.bcebos.com/TX2_JetPack4.3_opencv_3.4.10_gcc7.5.0.zip，并解压到deps文件夹下
OPENCV_DIR=/path/to/opencv

# 请检查以上各个路径是否正确

# 以下无需改动
cmake .. \
    -DWITH_GPU=${WITH_GPU} \
    -DWITH_MKL=OFF \
    -DWITH_TENSORRT=${WITH_TENSORRT} \
    -DTENSORRT_DIR=${TENSORRT_DIR} \
    -DPADDLE_DIR=${PADDLE_DIR} \
    -DWITH_STATIC_LIB=${WITH_STATIC_LIB} \
    -DCUDA_LIB=${CUDA_LIB} \
    -DCUDNN_LIB=${CUDNN_LIB} \
    -DOPENCV_DIR=${OPENCV_DIR} \
    -DPADDLE_LIB_NAME={PADDLE_LIB_NAME}
make
```

例如设置如下：
```
# 是否使用GPU(即是否使用 CUDA)
WITH_GPU=ON

# 是否使用MKL or openblas
WITH_MKL=OFF

# 是否集成 TensorRT(仅WITH_GPU=ON 有效)
WITH_TENSORRT=OFF

# TensorRT 的include路径
TENSORRT_INC_DIR=/usr/include/aarch64-linux-gnu

# TensorRT 的lib路径
TENSORRT_LIB_DIR=/usr/lib/aarch64-linux-gnu

# Paddle 预测库路径
PADDLE_DIR=/home/nvidia/PaddleDetection_infer/fluid_inference/

# Paddle 的预测库是否使用静态库来编译
# 使用TensorRT时，Paddle的预测库通常为动态库
WITH_STATIC_LIB=OFF

# CUDA 的 lib 路径
CUDA_LIB=/usr/local/cuda-10.0/lib64

# CUDNN 的 lib 路径
CUDNN_LIB=/usr/lib/aarch64-linux-gnu/
```

修改脚本设置好主要参数后，执行`build`脚本：
 ```shell
 sh ./scripts/build.sh
 ```

### Step5: 预测及可视化
编译成功后，预测入口程序为`build/main`其主要命令参数说明如下：
|  参数   | 说明  |
|  ----  | ----  |
| --model_dir  | 导出的预测模型所在路径 |
| --image_file  | 要预测的图片文件路径 |
| --video_path  | 要预测的视频文件路径 |
| --camera_id | Option | 用来预测的摄像头ID，默认为-1（表示不使用摄像头预测）|
| --device  | 运行时的设备，可选择`CPU/GPU/XPU`，默认为`CPU`|
| --gpu_id  |  指定进行推理的GPU device id(默认值为0)|
| --run_mode | 使用GPU时，默认为fluid, 可选（fluid/trt_fp32/trt_fp16/trt_int8）|
| --run_benchmark | 是否重复预测来进行benchmark测速 ｜
| --output_dir | 输出图片所在的文件夹, 默认为output ｜

**注意**: 如果同时设置了`video_path`和`image_file`，程序仅预测`video_path`。


`样例一`：
```shell
#不使用`GPU`测试图片 `/root/projects/images/test.jpeg`  
./main --model_dir=/root/projects/models/yolov3_darknet --image_path=/root/projects/images/test.jpeg
```

图片文件`可视化预测结果`会保存在当前目录下`output.jpg`文件中。


`样例二`:
```shell
#使用 `GPU`预测视频`/root/projects/videos/test.mp4`
./main --model_dir=/root/projects/models/yolov3_darknet --video_path=/root/projects/images/test.mp4 --device=GPU
```
视频文件目前支持`.mp4`格式的预测，`可视化预测结果`会保存在当前目录下`output.mp4`文件中。


## 性能测试
测试环境为：硬件: TX2，JetPack版本: 4.3, Paddle预测库: 1.8.4，CUDA: 10.0, CUDNN: 7.5, TensorRT: 5.0.  

去掉前100轮warmup时间，测试100轮的平均时间，单位ms/image，只计算模型运行时间，不包括数据的处理和拷贝。


|模型 | 输入| AnalysisPredictor(ms) |
|---|----|---|
| yolov3_mobilenet_v1 |  608*608  | 56.243858
| faster_rcnn_r50_1x  | 1333*1333  | 73.552460
| faster_rcnn_r50_vd_fpn_2x | 1344*1344 | 87.582146
| mask_rcnn_r50_fpn_1x | 1344*1344  | 107.317848
| mask_rcnn_r50_vd_fpn_2x | 1344*1344  | 87.98.708122
| ppyolo_r18vd | 320*320  |  22.876789
| ppyolo_2x | 608*608  | 68.562050
