# Visual Studio 2019 Community CMake 编译指南

Windows 平台下，我们使用`Visual Studio 2019 Community` 进行了测试。微软从`Visual Studio 2017`开始即支持直接管理`CMake`跨平台编译项目，但是直到`2019`才提供了稳定和完全的支持，所以如果你想使用CMake管理项目编译构建，我们推荐你使用`Visual Studio 2019`环境下构建。


## 前置条件
* Visual Studio 2019 (根据Paddle预测库所使用的VS版本选择，请参考 [Visual Studio 不同版本二进制兼容性](https://docs.microsoft.com/zh-cn/cpp/porting/binary-compat-2015-2017?view=vs-2019) )
* CUDA 9.0 / CUDA 10.0，cudnn 7+ （仅在使用GPU版本的预测库时需要）
* CMake 3.0+ [CMake下载](https://cmake.org/download/)

请确保系统已经安装好上述基本软件，我们使用的是`VS2019`的社区版。

**下面所有示例以工作目录为 `D:\projects`演示**。

### Step1: 下载代码

下载源代码
```shell
git clone https://github.com/PaddlePaddle/PaddleDetection.git
```

**说明**：其中`C++`预测代码在`PaddleDetection/deploy/cpp` 目录，该目录不依赖任何`PaddleDetection`下其他目录。


### Step2: 下载PaddlePaddle C++ 预测库 fluid_inference

PaddlePaddle C++ 预测库针对不同的`CPU`和`CUDA`版本提供了不同的预编译版本，请根据实际情况下载:  [C++预测库下载列表](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/05_inference_deployment/inference/windows_cpp_inference.html)

解压后`D:\projects\fluid_inference`目录包含内容为：
```
fluid_inference
├── paddle # paddle核心库和头文件
|
├── third_party # 第三方依赖库和头文件
|
└── version.txt # 版本和编译信息
```

### Step3: 安装配置OpenCV

1. 在OpenCV官网下载适用于Windows平台的3.4.6版本， [下载地址](https://sourceforge.net/projects/opencvlibrary/files/3.4.6/opencv-3.4.6-vc14_vc15.exe/download)  
2. 运行下载的可执行文件，将OpenCV解压至指定目录，如`D:\projects\opencv`
3. 配置环境变量，如下流程所示（如果使用全局绝对路径，可以不用设置环境变量）  
    - 我的电脑->属性->高级系统设置->环境变量
    - 在系统变量中找到Path（如没有，自行创建），并双击编辑
    - 新建，将opencv路径填入并保存，如`D:\projects\opencv\build\x64\vc14\bin`

### Step4: 编译

1. 进入到`cpp`文件夹
```
cd D:\projects\PaddleDetection\deploy\cpp
```

2. 使用CMake生成项目文件

编译参数的含义说明如下（带*表示仅在使用**GPU版本**预测库时指定, 其中CUDA库版本尽量对齐，**使用9.0、10.0版本，不使用9.2、10.1等版本CUDA库**）：

|  参数名   | 含义  |
|  ----  | ----  |
| *CUDA_LIB  | CUDA的库路径 |
| *CUDNN_LIB | CUDNN的库路径 |
| OPENCV_DIR  | OpenCV的安装路径， |
| PADDLE_DIR | Paddle预测库的路径 |
| PADDLE_LIB_NAME | Paddle 预测库名称 |

**注意：** 1. 使用`CPU`版预测库，请把`WITH_GPU`的勾去掉 2. 如果使用的是`openblas`版本，请把`WITH_MKL`勾去掉

执行如下命令项目文件：
```
cmake . -G "Visual Studio 16 2019" -A x64 -T host=x64 -DWITH_GPU=ON -DWITH_MKL=ON -DCMAKE_BUILD_TYPE=Release -DCUDA_LIB=path_to_cuda_lib -DCUDNN_LIB=path_to_cudnn_lib -DPADDLE_DIR=path_to_paddle_lib -DPADDLE_LIB_NAME=paddle_inference -DOPENCV_DIR=path_to_opencv
```

例如：
```
cmake . -G "Visual Studio 16 2019" -A x64 -T host=x64 -DWITH_GPU=ON -DWITH_MKL=ON -DCMAKE_BUILD_TYPE=Release -DCUDA_LIB=D:\projects\packages\cuda10_0\lib\x64 -DCUDNN_LIB=D:\projects\packages\cuda10_0\lib\x64 -DPADDLE_DIR=D:\projects\packages\fluid_inference -DPADDLE_LIB_NAME=paddle_inference -DOPENCV_DIR=D:\projects\packages\opencv3_4_6
```

3. 编译
用`Visual Studio 16 2019`打开`cpp`文件夹下的`PaddleObjectDetector.sln`，将编译模式设置为`Release`，点击`生成`->`全部生成


### Step5: 预测及可视化

上述`Visual Studio 2019`编译产出的可执行文件在`out\build\x64-Release`目录下，打开`cmd`，并切换到该目录：

```
cd D:\projects\PaddleDetection\deploy\cpp\out\build\x64-Release
```
可执行文件`main`即为样例的预测程序，其主要的命令行参数如下：

|  参数   | 说明  |
|  ----  | ----  |
| --model_dir  | 导出的预测模型所在路径 |
| --image_file  | 要预测的图片文件路径 |
| --video_path  | 要预测的视频文件路径 |
| --camera_id | Option | 用来预测的摄像头ID，默认为-1（表示不使用摄像头预测）|
| --device  | 运行时的设备，可选择`CPU/GPU/XPU`，默认为`CPU`|
| --gpu_id  |  指定进行推理的GPU device id(默认值为0)|
| --run_mode | 使用GPU时，默认为fluid, 可选（fluid/trt_fp32/trt_fp16/trt_int8）|
| --run_benchmark | 是否重复预测来进行benchmark测速 |
| --output_dir | 输出图片所在的文件夹, 默认为output |

**注意**：  
（1）如果同时设置了`video_path`和`image_file`，程序仅预测`video_path`。  
（2）如果提示找不到`opencv_world346.dll`，把`D:\projects\packages\opencv3_4_6\build\x64\vc14\bin`文件夹下的`opencv_world346.dll`拷贝到`main.exe`文件夹下即可。


`样例一`：
```shell
#不使用`GPU`测试图片 `D:\\images\\test.jpeg`  
.\main --model_dir=D:\\models\\yolov3_darknet --image_path=D:\\images\\test.jpeg
```

图片文件`可视化预测结果`会保存在当前目录下`output.jpg`文件中。


`样例二`:
```shell
#使用`GPU`测试视频 `D:\\videos\\test.mp4`  
.\main --model_dir=D:\\models\\yolov3_darknet --video_path=D:\\videos\\test.mp4 --device=GPU
```

视频文件目前支持`.mp4`格式的预测，`可视化预测结果`会保存在当前目录下`output.mp4`文件中。


## 性能测试
测试环境为：系统: Windows 10专业版系统，CPU: I9-9820X, GPU: GTX 2080 Ti，Paddle预测库: 1.8.4，CUDA: 10.0, CUDNN: 7.4.  

去掉前100轮warmup时间，测试100轮的平均时间，单位ms/image，只计算模型运行时间，不包括数据的处理和拷贝。


|模型 | AnalysisPredictor(ms) | 输入|
|---|----|---|
| YOLOv3-MobileNetv1 | 41.51 |  608*608
| faster_rcnn_r50_1x | 194.47 | 1333*1333
| faster_rcnn_r50_vd_fpn_2x | 43.35 | 1344*1344
| mask_rcnn_r50_fpn_1x | 96.96 | 1344*1344
| mask_rcnn_r50_vd_fpn_2x | 97.66 | 1344*1344
| ppyolo_r18vd | 5.54 | 320*320
| ppyolo_2x | 56.93 | 608*608
| ttfnet_darknet | 36.17 | 512*512
