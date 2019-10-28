# Visual Studio 2019 Community CMake 编译指南

Windows 平台下，我们使用`Visual Studio 2015` 和 `Visual Studio 2019 Community` 进行了测试。微软从`Visual Studio 2017`开始即支持直接管理`CMake`跨平台编译项目，但是直到`2019`才提供了稳定和完全的支持，所以如果你想使用CMake管理项目编译构建，我们推荐你使用`Visual Studio 2019`环境下构建。

你也可以使用和`VS2015`一样，通过把`CMake`项目转化成`VS`项目来编译，其中**有差别的部分**在文档中我们有说明，请参考：[使用Visual Studio 2015 编译指南](./windows_vs2015_build.md)

## 前置条件
* Visual Studio 2019
* CUDA 8.0/ CUDA 9.0
* CMake 3.0+

请确保系统已经安装好上述基本软件，我们使用的是`VS2019`的社区版。

**下面所有示例以工作目录为 `D:\projects`演示**。

### Step1: 下载代码

1. 点击下载源代码：[下载地址](https://github.com/PaddlePaddle/models/archive/develop.zip)
2. 解压，解压后目录重命名为`paddle_models`

以下代码目录路径为`D:\projects\paddle_models` 为例。


### Step2: 下载PaddlePaddle C++ 预测库 fluid_inference

根据Windows环境，下载相应版本的PaddlePaddle预测库，并解压到`D:\projects\`目录

| CUDA | GPU | 下载地址 |
|------|------|--------|
| 8.0 | Yes | [fluid_inference.zip](https://bj.bcebos.com/v1/paddleseg/fluid_inference_win.zip) |
| 9.0 | Yes | [fluid_inference_cuda90.zip](https://paddleseg.bj.bcebos.com/fluid_inference_cuda9_cudnn7.zip) |

解压后`D:\projects\fluid_inference`目录包含内容为：
```
fluid_inference
├── paddle # paddle核心库和头文件
|
├── third_party # 第三方依赖库和头文件
|
└── version.txt # 版本和编译信息
```
**注意：** `CUDA90`版本解压后目录名称为`fluid_inference_cuda90`。

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

4. 点击`浏览`，分别设置编译选项指定`CUDA`、`OpenCV`、`Paddle预测库`的路径

![step4](https://paddleseg.bj.bcebos.com/inference/vs2019_step5.png)

三个编译参数的含义说明如下：

|  参数名   | 含义  |
|  ----  | ----  |
| CUDA_LIB  | cuda的库路径 |
| OPENCV_DIR  | OpenCV的安装路径， |
| PADDLE_DIR | Paddle预测库的路径 |

**设置完成后**, 点击上图中`保存并生成CMake缓存以加载变量`。

5. 点击`生成`->`全部生成`

![step6](https://paddleseg.bj.bcebos.com/inference/vs2019_step6.png)


### Step5: 预测及可视化

上述`Visual Studio 2019`编译产出的可执行文件在`out\build\x64-Release`目录下，打开`cmd`，并切换到该目录：

```
cd D:\projects\paddle_models\models\PaddleCV\PaddleDetection\inference\build\x64-Release
```

之后执行命令：

```
detection_demo.exe --conf=/path/to/your/conf --input_dir=/path/to/your/input/data/directory
```

更详细说明请参考ReadMe文档： [预测和可视化部分](../README.md)

