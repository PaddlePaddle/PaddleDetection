# Windows平台使用 Visual Studio 2015 编译指南

本文档步骤，我们同时在`Visual Studio 2015` 和 `Visual Studio 2019 Community` 两个版本进行了测试，我们推荐使用[`Visual Studio 2019`直接编译`CMake`项目](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/static/deploy/cpp/docs/windows_vs2019_build.md)。


## 前置条件
* Visual Studio 2015
* CUDA 9.0 / CUDA 10.0，cudnn 7+ （仅在使用GPU版本的预测库时需要）
* CMake 3.0+

请确保系统已经安装好上述基本软件，**下面所有示例以工作目录为 `D:\projects`演示**。

### Step1: 下载代码

1. 打开`cmd`, 执行 `cd D:\projects`
2. `git clone https://github.com/PaddlePaddle/PaddleDetection.git`

`C++`预测库代码在`D:\projects\PaddleDetection\inference` 目录，该目录不依赖任何`PaddleDetection`下其他目录。


### Step2: 下载PaddlePaddle C++ 预测库 fluid_inference

PaddlePaddle C++ 预测库主要分为两大版本：CPU版本和GPU版本。其中，针对不同的CUDA版本，GPU版本预测库又分为两个版本预测库：CUDA 9.0和CUDA 10.0版本预测库。根据Windows环境，下载相应版本的PaddlePaddle预测库，并解压到`D:\projects\`目录。以下为各版本C++预测库的下载链接：

|  版本   | 链接  |
|  ----  | ----  |
| CPU版本  | [fluid_inference_install_dir.zip](https://bj.bcebos.com/paddlehub/paddle_inference_lib/fluid_install_dir_win_cpu_1.6.zip) |
| CUDA 9.0版本  | [fluid_inference_install_dir.zip](https://bj.bcebos.com/paddlehub/paddle_inference_lib/fluid_inference_install_dir_win_cuda9_1.6.1.zip) |
| CUDA 10.0版本  | [fluid_inference_install_dir.zip](https://bj.bcebos.com/paddlehub/paddle_inference_lib/fluid_inference_install_dir_win_cuda10_1.6.1.zip) |

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

### Step4: 以VS2015为例编译代码

以下命令需根据自己系统中各相关依赖的路径进行修改

* 调用VS2015, 请根据实际VS安装路径进行调整，打开cmd命令行工具执行以下命令
* 其他vs版本(比如vs2019)，请查找到对应版本的`vcvarsall.bat`路径，替换本命令即可

```
call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" amd64
```

三个编译参数的含义说明如下（带*表示仅在使用**GPU版本**预测库时指定, 其中CUDA库版本尽量对齐，**使用9.0、10.0版本，不使用9.2、10.1等版本CUDA库**）：

|  参数名   | 含义  |
|  ----  | ----  |
| *CUDA_LIB  | CUDA的库路径 |
| OPENCV_DIR  | OpenCV的安装路径， |
| PADDLE_DIR | Paddle预测库的路径 |

在使用**GPU版本**预测库进行编译时，可执行下列操作。**注意**把对应的参数改为你的上述依赖库实际路径：

```bash
# 切换到预测库所在目录
cd /d D:\projects\PaddleDetection\inference
# 创建构建目录, 重新构建只需要删除该目录即可
mkdir build
cd build
# cmake构建VS项目
D:\projects\PaddleDetection\inference\build> cmake .. -G "Visual Studio 14 2015 Win64" -DWITH_GPU=ON -DPADDLE_DIR=D:\projects\fluid_inference -DCUDA_LIB=D:\projects\cudalib\v9.0\lib\x64 -DOPENCV_DIR=D:\projects\opencv -T host=x64
```

在使用**CPU版本**预测库进行编译时，可执行下列操作：
```bash
# 切换到预测库所在目录
cd /d D:\projects\PaddleDetection\inference
# 创建构建目录, 重新构建只需要删除该目录即可
mkdir build
cd build
# cmake构建VS项目
D:\projects\PaddleDetection\inference\build> cmake .. -G "Visual Studio 14 2015 Win64" -DWITH_GPU=OFF -DPADDLE_DIR=D:\projects\fluid_inference -DOPENCV_DIR=D:\projects\opencv -T host=x64
```


这里的`cmake`参数`-G`, 表示生成对应的VS版本的工程，可以根据自己的`VS`版本调整，具体请参考[cmake文档](https://cmake.org/cmake/help/v3.15/manual/cmake-generators.7.html)

* 生成可执行文件

```
D:\projects\PaddleDetection\inference\build> msbuild /m /p:Configuration=Release cpp_inference_demo.sln
```

### Step5: 预测及可视化

上述`Visual Studio 2015`编译产出的可执行文件在`build\release`目录下，切换到该目录：
```
cd /d D:\projects\PaddleDetection\inference\build\release
```

之后执行命令：

```
detection_demo.exe --conf=/path/to/your/conf --input_dir=/path/to/your/input/data/directory
```

更详细说明请参考ReadMe文档： [预测和可视化部分](../../deploy/README.md)
