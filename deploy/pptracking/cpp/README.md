# C++端预测部署

在PaddlePaddle中预测引擎和训练引擎底层有着不同的优化方法, 预测引擎使用了AnalysisPredictor，专门针对推理进行了优化，该引擎可以对模型进行多项图优化，减少不必要的内存拷贝。如果用户在部署已训练模型的过程中对性能有较高的要求，我们提供了独立于PaddleDetection的预测脚本，方便用户直接集成部署。当前C++部署支持基于Fairmot的单镜头类别预测部署，并支持人流量统计、出入口计数功能。

主要包含三个步骤：
- 准备环境
- 导出预测模型
- C++预测

## 一、准备环境

环境要求：

- GCC 8.2
- CUDA 10.1/10.2/11.1; CUDNN 7.6/8.1
- CMake 3.0+
- TensorRT 6/7

NVIDIA Jetson用户请参考[Jetson平台编译指南](../../cpp/docs/Jetson_build.md#jetson环境搭建)完成JetPack安装

### 1. 下载代码

```
git clone https://github.com/PaddlePaddle/PaddleDetection.git
# C++部署代码与其他目录代码独立
cd deploy/pptracking/cpp
```

### 2. 下载或编译PaddlePaddle C++预测库

请根据环境选择适当的预测库进行下载，参考[C++预测库下载列表](https://paddleinference.paddlepaddle.org.cn/user_guides/download_lib.html)

下载并解压后`./paddle_inference`目录包含内容为：

```
paddle_inference
├── paddle # paddle核心库和头文件
|
├── third_party # 第三方依赖库和头文件
|
└── version.txt # 版本和编译信息
```

**注意:** 如果用户环境与官网提供环境不一致（如cuda 、cudnn、tensorrt版本不一致等），或对飞桨源代码有修改需求，或希望进行定制化构建，可参考[文档](https://paddleinference.paddlepaddle.org.cn/user_guides/source_compile.html)自行源码编译预测库。

### 3. 编译

<details>
<summary>Linux编译:</summary>

编译`cmake`的命令在`scripts/build.sh`中，请根据实际情况修改主要参数，其主要内容说明如下：

```
# 是否使用GPU(即是否使用 CUDA)
WITH_GPU=ON

# 是否使用MKL or openblas，TX2需要设置为OFF
WITH_MKL=OFF

# 是否集成 TensorRT(仅WITH_GPU=ON 有效)
WITH_TENSORRT=ON

# TensorRT 的include路径
TENSORRT_INC_DIR=/path/to/TensorRT/include

# TensorRT 的lib路径
TENSORRT_LIB_DIR=/path/to/TensorRT/lib

# Paddle 预测库路径
PADDLE_DIR=/path/to/paddle_inference/

# Paddle 预测库名称
PADDLE_LIB_NAME=libpaddle_inference

# CUDA 的 lib 路径
CUDA_LIB=/path/to/cuda/lib

# CUDNN 的 lib 路径
CUDNN_LIB=/path/to/cudnn/lib

# OPENCV路径
OPENCV_DIR=/path/to/opencv
```

修改脚本设置好主要参数后，执行```build.sh```脚本：

```
sh ./scripts/build.sh
```


</details>
<details>
<summary>Windows编译:</summary>

- 安装配置OpenCV
 1. 在OpenCV官网下载适用于Windows平台的3.4.6版本，[下载地址](https://sourceforge.net/projects/opencvlibrary/files/3.4.6/opencv-3.4.6-vc14_vc15.exe/download)  
 2. 运行下载的可执行文件，将OpenCV解压至指定目录，如`D:\projects\opencv`
 3. 配置环境变量，如下流程所示（如果使用全局绝对路径，可以不用设置环境变量）  

    - 我的电脑->属性->高级系统设置->环境变量
    - 在系统变量中找到Path（如没有，自行创建），并双击编辑
    - 新建，将opencv路径填入并保存，如`D:\projects\opencv\build\x64\vc14\bin`

- 使用CMake生成项目文件

	执行如下命令项目文件：
```
cmake . -G "Visual Studio 16 2019" -A x64 -T host=x64 -DWITH_GPU=ON -DWITH_MKL=ON -DCMAKE_BUILD_TYPE=Release -DCUDA_LIB=path_to_cuda_lib -DCUDNN_LIB=path_to_cudnn_lib -DPADDLE_DIR=path_to_paddle_lib -DPADDLE_LIB_NAME=paddle_inference -DOPENCV_DIR=path_to_opencv -DWITH_KEYPOINT=ON
```

- 编译
用`Visual Studio 2019`打开`cpp`文件夹下的`PaddleObjectDetector.sln`，将编译模式设置为`Release`，点击`生成`->`全部生成

编译产出的可执行文件在`Release`目录下

</details>

**注意：**

1. `TX2`平台的`CUDA`、`CUDNN`需要通过`JetPack`安装。
2. 已提供linux和tx2平台的opencv下载方式，其他环境请自行安装[opencv](https://opencv.org/)
3. Windows用户推荐使用Visual Studio 2019编译

## 二、导出预测模型

将训练保存的权重导出为预测库需要的模型格式，使用PaddleDetection下的```tools/export_model.py```导出模型

```
python tools/export_model.py -c configs/mot/fairmot/fairmot_hrnetv2_w18_dlafpn_30e_576x320.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/fairmot_hrnetv2_w18_dlafpn_30e_576x320.pdparams
```

预测模型会默认导出到```output_inference/fairmot_hrnetv2_w18_dlafpn_30e_576x320```目录下，包括```infer_cfg.yml```, ```model.pdiparams```, ```model.pdiparams.info```, ```model.pdmodel```

导出模型也可以通过[预测模型列表](../README.md)中'算法介绍部分'直接下载使用

## 三、C++预测

完成以上步骤后，可以通过```build/main```(Linux)或```main.exe```(Windows)进行预测，参数列表如下:

|  参数   | 说明  |
|  ----  | ----  |
| --track_model_dir  | 导出的跟踪预测模型所在路径 |
| --video_file  | 要预测的视频文件路径 |
| --device  | 运行时的设备，可选择`CPU/GPU/XPU`，默认为`CPU`|
| --gpu_id  |  指定进行推理的GPU device id(默认值为0)|
| --run_mode | 使用GPU时，默认为paddle, 可选（paddle/trt_fp32/trt_fp16/trt_int8）|
| --output_dir | 输出图片所在的文件夹, 默认为output ｜
| --use_mkldnn | CPU预测中是否开启MKLDNN加速 |
| --cpu_threads | 设置cpu线程数，默认为1 |
| --do_entrance_counting | 是否进行出入口流量统计，默认为否 |
| --save_result | 是否保存跟踪结果 |

样例一：

```shell
# 使用CPU测试视频 `test.mp4` , 模型和测试视频均移至`build`目录下

./main --track_model_dir=./fairmot_hrnetv2_w18_dlafpn_30e_576x320 --video_file=test.mp4

# 视频可视化预测结果默认保存在当前目录下output/test.mp4文件中
```


样例二：

```shell
# 使用GPU测试视频 `test.mp4` , 模型和测试视频均移至`build`目录下，实现出入口计数功能，并保存跟踪结果

./main -video_file=test.mp4 -track_model_dir=./fairmot_dla34_30e_1088x608/  --device=gpu --do_entrance_counting=True --save_result=True

# 视频可视化预测结果默认保存在当前目录下`output/test.mp4`中
# 跟踪结果保存在`output/mot_output.txt`中
# 计数结果保存在`output/flow_statistic.txt`中
```
