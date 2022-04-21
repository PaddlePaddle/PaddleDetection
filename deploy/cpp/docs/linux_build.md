# Linux平台编译指南

## 说明
本文档在 `Linux`平台使用`GCC 8.2`测试过，如果需要使用其他G++版本编译使用，则需要重新编译Paddle预测库，请参考: [从源码编译Paddle预测库](https://paddleinference.paddlepaddle.org.cn/user_guides/source_compile.html)。本文档使用的预置的opencv库是在ubuntu 16.04上用gcc8.2编译的，如果需要在gcc8.2以外的环境编译，那么需自行编译opencv库。

## 前置条件
* G++ 8.2
* CUDA 9.0 / CUDA 10.1, cudnn 7+ （仅在使用GPU版本的预测库时需要）
* CMake 3.0+

请确保系统已经安装好上述基本软件，**下面所有示例以工作目录为 `/root/projects/`演示**。

### Step1: 下载代码

 `git clone https://github.com/PaddlePaddle/PaddleDetection.git`

**说明**：其中`C++`预测代码在`/root/projects/PaddleDetection/deploy/cpp` 目录，该目录不依赖任何`PaddleDetection`下其他目录。


### Step2: 下载PaddlePaddle C++ 预测库 paddle_inference

PaddlePaddle C++ 预测库针对不同的`CPU`和`CUDA`版本提供了不同的预编译版本，请根据实际情况下载:  [C++预测库下载列表](https://paddleinference.paddlepaddle.org.cn/user_guides/download_lib.html)


下载并解压后`/root/projects/paddle_inference`目录包含内容为：
```
paddle_inference
├── paddle # paddle核心库和头文件
|
├── third_party # 第三方依赖库和头文件
|
└── version.txt # 版本和编译信息
```

**注意:** 预编译版本除`nv-jetson-cuda10-cudnn7.5-trt5` 以外其它包都是基于`GCC 4.8.5`编译，使用高版本`GCC`可能存在 `ABI`兼容性问题，建议降级或[自行编译预测库](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/inference/build_and_install_lib_cn.html)。


### Step3: 编译

编译`cmake`的命令在`scripts/build.sh`中，请根据实际情况修改主要参数，其主要内容说明如下：

```
# 是否使用GPU(即是否使用 CUDA)
WITH_GPU=OFF

# 使用MKL or openblas
WITH_MKL=ON

# 是否集成 TensorRT(仅WITH_GPU=ON 有效)
WITH_TENSORRT=OFF

# TensorRT 的include路径
TENSORRT_LIB_DIR=/path/to/TensorRT/include

# TensorRT 的lib路径
TENSORRT_LIB_DIR=/path/to/TensorRT/lib

# Paddle 预测库路径
PADDLE_DIR=/path/to/paddle_inference

# Paddle 预测库名称
PADDLE_LIB_NAME=paddle_inference

# CUDA 的 lib 路径
CUDA_LIB=/path/to/cuda/lib

# CUDNN 的 lib 路径
CUDNN_LIB=/path/to/cudnn/lib

# 是否开启关键点模型预测功能
WITH_KEYPOINT=ON

# 请检查以上各个路径是否正确

# 以下无需改动
cmake .. \
    -DWITH_GPU=${WITH_GPU} \
    -DWITH_MKL=${WITH_MKL} \
    -DWITH_TENSORRT=${WITH_TENSORRT} \
    -DTENSORRT_LIB_DIR=${TENSORRT_LIB_DIR} \
    -DTENSORRT_INC_DIR=${TENSORRT_INC_DIR} \
    -DPADDLE_DIR=${PADDLE_DIR} \
    -DCUDA_LIB=${CUDA_LIB} \
    -DCUDNN_LIB=${CUDNN_LIB} \
    -DOPENCV_DIR=${OPENCV_DIR} \
    -DPADDLE_LIB_NAME=${PADDLE_LIB_NAME} \
    -DWITH_KEYPOINT=${WITH_KEYPOINT}
make

```

修改脚本设置好主要参数后，执行`build`脚本：
 ```shell
 sh ./scripts/build.sh
 ```

**注意**: OPENCV依赖OPENBLAS，Ubuntu用户需确认系统是否已存在`libopenblas.so`。如未安装，可执行apt-get install libopenblas-dev进行安装。

### Step4: 预测及可视化
编译成功后，预测入口程序为`build/main`其主要命令参数说明如下：
|  参数   | 说明  |
|  ----  | ----  |
| --model_dir  | 导出的检测预测模型所在路径 |
| --model_dir_keypoint  | Option | 导出的关键点预测模型所在路径 |
| --image_file  | 要预测的图片文件路径 |
| --image_dir  |  要预测的图片文件夹路径   |
| --video_file  | 要预测的视频文件路径 |
| --camera_id | Option | 用来预测的摄像头ID，默认为-1（表示不使用摄像头预测）|
| --device  | 运行时的设备，可选择`CPU/GPU/XPU`，默认为`CPU`|
| --gpu_id  |  指定进行推理的GPU device id(默认值为0)|
| --run_mode | 使用GPU时，默认为paddle, 可选（paddle/trt_fp32/trt_fp16/trt_int8）|
| --batch_size  | 检测模型预测时的batch size，在指定`image_dir`时有效 |
| --batch_size_keypoint  | 关键点模型预测时的batch size，默认为8 |
| --run_benchmark | 是否重复预测来进行benchmark测速 ｜
| --output_dir | 输出图片所在的文件夹, 默认为output ｜
| --use_mkldnn | CPU预测中是否开启MKLDNN加速 |
| --cpu_threads | 设置cpu线程数，默认为1 |
| --use_dark | 关键点模型输出预测是否使用DarkPose后处理，默认为true |

**注意**:
- 优先级顺序：`camera_id` > `video_file` > `image_dir` > `image_file`。
- --run_benchmark如果设置为True，则需要安装依赖`pip install pynvml psutil GPUtil`。

`样例一`：
```shell
#不使用`GPU`测试图片 `/root/projects/images/test.jpeg`  
./build/main --model_dir=/root/projects/models/yolov3_darknet --image_file=/root/projects/images/test.jpeg
```

图片文件`可视化预测结果`会保存在当前目录下`output.jpg`文件中。


`样例二`:
```shell
#使用 `GPU`预测视频`/root/projects/videos/test.mp4`
./build/main --model_dir=/root/projects/models/yolov3_darknet --video_file=/root/projects/images/test.mp4 --device=GPU
```
视频文件目前支持`.mp4`格式的预测，`可视化预测结果`会保存在当前目录下`output.mp4`文件中。


`样例三`：
```shell
#使用关键点模型与检测模型联合预测，使用 `GPU`预测  
#检测模型检测到的人送入关键点模型进行关键点预测
./build/main --model_dir=/root/projects/models/yolov3_darknet --model_dir_keypoint=/root/projects/models/hrnet_w32_256x192 --image_file=/root/projects/images/test.jpeg --device=GPU
```

## 性能测试
benchmark请查看[BENCHMARK_INFER](../../BENCHMARK_INFER.md)
