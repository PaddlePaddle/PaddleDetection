# Visual Studio 2019 Community CMake 编译指南

Windows 平台下，我们使用`Visual Studio 2019 Community` 进行了测试。微软从`Visual Studio 2017`开始即支持直接管理`CMake`跨平台编译项目，但是直到`2019`才提供了稳定和完全的支持，所以如果你想使用CMake管理项目编译构建，我们推荐你使用`Visual Studio 2019`环境下构建。


## 前置条件
* Visual Studio 2019
* CUDA 9.0 / CUDA 10.0，cudnn 7+ （仅在使用GPU版本的预测库时需要）
* CMake 3.0+

请确保系统已经安装好上述基本软件，我们使用的是`VS2019`的社区版。

**下面所有示例以工作目录为 `D:\projects`演示**。

### Step1: 下载代码

下载源代码
```shell
git clone https://github.com/PaddlePaddle/PaddleDetection.git
```

**说明**：其中`C++`预测代码在`PaddleDetection/deploy/cpp` 目录，该目录不依赖任何`PaddleDetection`下其他目录。


### Step2: 下载PaddlePaddle C++ 预测库 fluid_inference

PaddlePaddle C++ 预测库针对不同的`CPU`和`CUDA`版本提供了不同的预编译版本，请根据实际情况下载:  [C++预测库下载列表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/advanced_guide/inference_deployment/inference/windows_cpp_inference.html)

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
3. 配置环境变量，如下流程所示  
    - 我的电脑->属性->高级系统设置->环境变量
    - 在系统变量中找到Path（如没有，自行创建），并双击编辑
    - 新建，将opencv路径填入并保存，如`D:\projects\opencv\build\x64\vc14\bin`

### Step4: 使用Visual Studio 2019直接编译CMake

1. 打开Visual Studio 2019 Community，点击`继续但无需代码`
![step2](https://paddleseg.bj.bcebos.com/inference/vs2019_step1.png)
2. 点击： `文件`->`打开`->`CMake`
![step2.1](https://paddleseg.bj.bcebos.com/inference/vs2019_step2.png)

选择项目代码所在路径，并打开`CMakeList.txt`：

![step2.2](https://paddleseg.bj.bcebos.com/inference/vs2019_step3.png)

3. 点击：`项目`->`cpp_inference_demo的CMake设置`

![step3](https://paddleseg.bj.bcebos.com/inference/vs2019_step4.png)

4. 点击`浏览`，分别设置编译选项指定`CUDA`、`CUDNN_LIB`、`OpenCV`、`Paddle预测库`的路径

三个编译参数的含义说明如下（带*表示仅在使用**GPU版本**预测库时指定, 其中CUDA库版本尽量对齐，**使用9.0、10.0版本，不使用9.2、10.1等版本CUDA库**）：

|  参数名   | 含义  |
|  ----  | ----  |
| *CUDA_LIB  | CUDA的库路径 |
| CUDNN_LIB | CUDNN的库路径 |
| OPENCV_DIR  | OpenCV的安装路径， |
| PADDLE_DIR | Paddle预测库的路径 |

**注意：** 1. 使用`CPU`版预测库，请把`WITH_GPU`的勾去掉 2. 如果使用的是`openblas`版本，请把`WITH_MKL`勾去掉
![step4](https://paddleseg.bj.bcebos.com/inference/vs2019_step5.png)

**设置完成后**, 点击上图中`保存并生成CMake缓存以加载变量`。

5. 点击`生成`->`全部生成`

![step6](https://paddleseg.bj.bcebos.com/inference/vs2019_step6.png)


### Step5: 预测及可视化

上述`Visual Studio 2019`编译产出的可执行文件在`out\build\x64-Release`目录下，打开`cmd`，并切换到该目录：

```
cd D:\projects\PaddleDetection\deploy\cpp\out\build\x64-Release
```
可执行文件`main`即为样例的预测程序，其主要的命令行参数如下：

|  参数   | 说明  |
|  ----  | ----  |
| model_dir  | 导出的预测模型所在路径 |
| image_path  | 要预测的图片文件路径 |
| video_path  | 要预测的视频文件路径 |
| use_gpu  | 是否使用 GPU 预测, 支持值为0或1(默认值为0)|
| gpu_id  |  指定进行推理的GPU device id(默认值为0)|

**注意**：如果同时设置了`video_path`和`image_path`，程序仅预测`video_path`。


`样例一`：
```shell
#不使用`GPU`测试图片 `D:\\images\\test.jpeg`  
.\main --model_dir=D:\\models\\yolov3_darknet --image_path=D:\\images\\test.jpeg
```

图片文件`可视化预测结果`会保存在当前目录下`output.jpeg`文件中。


`样例二`:
```shell
#使用`GPU`测试视频 `D:\\videos\\test.mp4`  
.\main --model_dir=D:\\models\\yolov3_darknet --video_path=D:\\videos\\test.mp4 --use_gpu=1
```

视频文件目前支持`.mp4`格式的预测，`可视化预测结果`会保存在当前目录下`output.mp4`文件中。
