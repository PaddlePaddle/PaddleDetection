# Visual Studio 2019 Community CMake 编译指南

Windows 平台下，我们使用`Visual Studio 2019 Community` 进行了测试。微软从`Visual Studio 2017`开始即支持直接管理`CMake`跨平台编译项目，但是直到`2019`才提供了稳定和完全的支持，所以如果你想使用CMake管理项目编译构建，我们推荐你使用`Visual Studio 2019`环境下构建。


## 前置条件
* Visual Studio 2019 (根据Paddle预测库所使用的VS版本选择，请参考 [Visual Studio 不同版本二进制兼容性](https://docs.microsoft.com/zh-cn/cpp/porting/binary-compat-2015-2017?view=vs-2019) )
* CUDA 9.0 / CUDA 10.0，cudnn 7+ / TensorRT（仅在使用GPU版本的预测库时需要）
* CMake 3.0+ [CMake下载](https://cmake.org/download/)

**特别注意：windows下预测库需要的TensorRT版本为：**。

|  预测库版本   | TensorRT版本  |
|  ----  | ----  |
| cuda10.1_cudnn7.6_avx_mkl_trt6 |  TensorRT-6.0.1.5  |
| cuda10.2_cudnn7.6_avx_mkl_trt7 |  TensorRT-7.0.0.11 |
| cuda11.0_cudnn8.0_avx_mkl_trt7 |  TensorRT-7.2.1.6  |

请确保系统已经安装好上述基本软件，我们使用的是`VS2019`的社区版。

**下面所有示例以工作目录为 `D:\projects`演示**。

### Step1: 下载代码

下载源代码
```shell
git clone https://github.com/PaddlePaddle/PaddleDetection.git
```

**说明**：其中`C++`预测代码在`PaddleDetection/deploy/cpp` 目录，该目录不依赖任何`PaddleDetection`下其他目录。


### Step2: 下载PaddlePaddle C++ 预测库 paddle_inference

PaddlePaddle C++ 预测库针对不同的`CPU`和`CUDA`版本提供了不同的预编译版本，请根据实际情况下载:  [C++预测库下载列表](https://paddleinference.paddlepaddle.org.cn/user_guides/download_lib.html#windows)

解压后`D:\projects\paddle_inference`目录包含内容为：
```
paddle_inference
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

编译参数的含义说明如下（带`*`表示仅在使用**GPU版本**预测库时指定, 其中CUDA库版本尽量对齐，**使用9.0、10.0版本，不使用9.2、10.1等版本CUDA库**）：

|  参数名   | 含义  |
|  ----  | ----  |
| *CUDA_LIB  | CUDA的库路径 |
| *CUDNN_LIB | CUDNN的库路径 |
| OPENCV_DIR  | OpenCV的安装路径， |
| PADDLE_DIR | Paddle预测库的路径 |
| PADDLE_LIB_NAME | Paddle 预测库名称 |

**注意：**

1. 如果编译环境为CPU，需要下载`CPU`版预测库，请把`WITH_GPU`的勾去掉
2. 如果使用的是`openblas`版本，请把`WITH_MKL`勾去掉
3. 如无需使用关键点模型可以把`WITH_KEYPOINT`勾去掉
4. Windows环境下，`PADDLE_LIB_NAME`需要设置为`paddle_inference`

执行如下命令项目文件：
```
cmake . -G "Visual Studio 16 2019" -A x64 -T host=x64 -DWITH_GPU=ON -DWITH_MKL=ON -DCMAKE_BUILD_TYPE=Release -DCUDA_LIB=path_to_cuda_lib -DCUDNN_LIB=path_to_cudnn_lib -DPADDLE_DIR=path_to_paddle_lib -DPADDLE_LIB_NAME=paddle_inference -DOPENCV_DIR=path_to_opencv -DWITH_KEYPOINT=ON
```

例如：
```
cmake . -G "Visual Studio 16 2019" -A x64 -T host=x64 -DWITH_GPU=ON -DWITH_MKL=ON -DCMAKE_BUILD_TYPE=Release -DCUDA_LIB=D:\projects\packages\cuda10_0\lib\x64 -DCUDNN_LIB=D:\projects\packages\cuda10_0\lib\x64 -DPADDLE_DIR=D:\projects\packages\paddle_inference -DPADDLE_LIB_NAME=paddle_inference -DOPENCV_DIR=D:\projects\packages\opencv3_4_6 -DWITH_KEYPOINT=ON
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

**注意**：  
（1）优先级顺序：`camera_id` > `video_file` > `image_dir` > `image_file`。
（2）如果提示找不到`opencv_world346.dll`，把`D:\projects\packages\opencv3_4_6\build\x64\vc14\bin`文件夹下的`opencv_world346.dll`拷贝到`main.exe`文件夹下即可。
（3）--run_benchmark如果设置为True，则需要安装依赖`pip install pynvml psutil GPUtil`。


`样例一`：
```shell
#不使用`GPU`测试图片 `D:\\images\\test.jpeg`  
.\main --model_dir=D:\\models\\yolov3_darknet --image_file=D:\\images\\test.jpeg
```

图片文件`可视化预测结果`会保存在当前目录下`output.jpg`文件中。


`样例二`:
```shell
#使用`GPU`测试视频 `D:\\videos\\test.mp4`  
.\main --model_dir=D:\\models\\yolov3_darknet --video_path=D:\\videos\\test.mp4 --device=GPU
```

视频文件目前支持`.mp4`格式的预测，`可视化预测结果`会保存在当前目录下`output.mp4`文件中。


`样例三`：
```shell
#使用关键点模型与检测模型联合预测，使用 `GPU`预测  
#检测模型检测到的人送入关键点模型进行关键点预测
.\main --model_dir=D:\\models\\yolov3_darknet --model_dir_keypoint=D:\\models\\hrnet_w32_256x192 --image_file=D:\\images\\test.jpeg --device=GPU
```

## 性能测试
Benchmark请查看[BENCHMARK_INFER](../../BENCHMARK_INFER.md)
