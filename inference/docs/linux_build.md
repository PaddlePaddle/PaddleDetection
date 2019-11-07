# Linux平台 编译指南

## 说明
本文档在 `Linux`平台使用`GCC 4.8.5` 和 `GCC 4.9.4`测试过，如果需要使用更高G++版本编译使用，则需要重新编译Paddle预测库，请参考: [从源码编译Paddle预测库](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/advanced_usage/deploy/inference/build_and_install_lib_cn.html#id15)。

## 前置条件
* G++ 4.8.2 ~ 4.9.4
* CUDA 9.0 / CUDA 10.0, cudnn 7+ （仅在使用GPU版本的预测库时需要）
* CMake 3.0+

请确保系统已经安装好上述基本软件，**下面所有示例以工作目录为 `/root/projects/`演示**。

### Step1: 下载代码

1. `git clone https://github.com/PaddlePaddle/PaddleDetection.git`

`C++`预测代码在`/root/projects/PaddleDetection/inference` 目录，该目录不依赖任何`PaddleDetection`下其他目录。


### Step2: 下载PaddlePaddle C++ 预测库 fluid_inference

PaddlePaddle C++ 预测库主要分为CPU版本和GPU版本。其中，针对不同的CUDA版本，GPU版本预测库又分为两个版本预测库：CUDA 9.0和CUDA 10.0版本预测库。以下为各版本C++预测库的下载链接：

|  版本   | 链接  |
|  ----  | ----  |
| CPU版本  | [fluid_inference.tgz](https://bj.bcebos.com/paddlehub/paddle_inference_lib/fluid_inference_linux_cpu_1.6.1.tgz) |
| CUDA 9.0版本  | [fluid_inference.tgz](https://bj.bcebos.com/paddlehub/paddle_inference_lib/fluid_inference_linux_cuda97_1.6.1.tgz) |
| CUDA 10.0版本  | [fluid_inference.tgz](https://bj.bcebos.com/paddlehub/paddle_inference_lib/fluid_inference_linux_cuda10_1.6.1.tgz) |


针对不同的CPU类型、不同的指令集，官方提供更多可用的预测库版本，目前已经推出1.6版本的预测库。其余版本具体请参考以下链接:[C++预测库下载列表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/advanced_usage/deploy/inference/build_and_install_lib_cn.html)


下载并解压后`/root/projects/fluid_inference`目录包含内容为：
```
fluid_inference
├── paddle # paddle核心库和头文件
|
├── third_party # 第三方依赖库和头文件
|
└── version.txt # 版本和编译信息
```

### Step3: 安装配置OpenCV

```shell
# 0. 切换到/root/projects目录
cd /root/projects
# 1. 下载OpenCV3.4.6版本源代码
wget -c https://paddleseg.bj.bcebos.com/inference/opencv-3.4.6.zip
# 2. 解压
unzip opencv-3.4.6.zip && cd opencv-3.4.6
# 3. 创建build目录并编译, 这里安装到/usr/local/opencv3目录
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/root/projects/opencv3 -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DWITH_IPP=OFF -DBUILD_IPP_IW=OFF -DWITH_LAPACK=OFF -DWITH_EIGEN=OFF -DCMAKE_INSTALL_LIBDIR=lib64 -DWITH_ZLIB=ON -DBUILD_ZLIB=ON -DWITH_JPEG=ON -DBUILD_JPEG=ON -DWITH_PNG=ON -DBUILD_PNG=ON -DWITH_TIFF=ON -DBUILD_TIFF=ON
make -j4
make install
```

**注意：** 上述操作完成后，`opencv` 被安装在 `/root/projects/opencv3` 目录。

### Step4: 编译

`CMake`编译时，涉及到四个编译参数用于指定核心依赖库的路径, 他们的定义如下:（带*表示仅在使用**GPU版本**预测库时指定，其中CUDA库版本尽量对齐，**使用9.0、10.0版本，不使用9.2、10.1版本CUDA库**）

|  参数名   | 含义  |
|  ----  | ----  |
| * CUDA_LIB  | CUDA的库路径 |
| * CUDNN_LIB | cudnn的库路径|
| OPENCV_DIR  | OpenCV的安装路径 |
| PADDLE_DIR | Paddle预测库的路径 |

在使用**GPU版本**预测库进行编译时，可执行下列操作。**注意**把对应的参数改为你的上述依赖库实际路径：

```shell
cd /root/projects/PaddleDetection/inference
mkdir build && cd build
cmake .. -DWITH_GPU=ON  -DPADDLE_DIR=/root/projects/fluid_inference -DCUDA_LIB=/usr/local/cuda/lib64/ -DOPENCV_DIR=/root/projects/opencv3/ -DCUDNN_LIB=/usr/local/cuda/lib64/ -DWITH_STATIC_LIB=OFF
make
```

在使用**CPU版本**预测库进行编译时，可执行下列操作：

```shell
cd /root/projects/PaddleDetection/inference

mkdir build && cd build
cmake .. -DWITH_GPU=OFF  -DPADDLE_DIR=/root/projects/fluid_inference -DOPENCV_DIR=/root/projects/opencv3/ -DWITH_STATIC_LIB=OFF
make
```

### Step5: 预测及可视化

执行命令：

```
./detection_demo --conf=/path/to/your/conf --input_dir=/path/to/your/input/data/directory
```

更详细说明请参考ReadMe文档： [预测和可视化部分](../README.md)
