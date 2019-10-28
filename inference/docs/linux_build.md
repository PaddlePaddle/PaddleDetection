# Linux平台 编译指南

## 说明
本文档在 `Linux`平台使用`GCC 4.8.5` 和 `GCC 4.9.4`测试过，如果需要使用更高G++版本编译使用，则需要重新编译Paddle预测库，请参考: [从源码编译Paddle预测库](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/advanced_usage/deploy/inference/build_and_install_lib_cn.html#id15)。

## 前置条件
* G++ 4.8.2 ~ 4.9.4
* CUDA 8.0/ CUDA 9.0
* CMake 3.0+

请确保系统已经安装好上述基本软件，**下面所有示例以工作目录为 `/root/projects/`演示**。

### Step1: 下载代码

1. `mkdir -p /root/projects/paddle_models && cd /root/projects/paddle_models`
2. `git clone https://github.com/PaddlePaddle/models.git`

`C++`预测代码在`/root/projects/paddle_models/models/PaddleCV/PaddleDetection/inference` 目录，该目录不依赖任何`PaddleDetection`下其他目录。


### Step2: 下载PaddlePaddle C++ 预测库 fluid_inference

目前仅支持`CUDA 8` 和 `CUDA 9`，请点击 [PaddlePaddle预测库下载地址](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/advanced_usage/deploy/inference/build_and_install_lib_cn.html)下载对应的版本（develop版本）。


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

`CMake`编译时，涉及到四个编译参数用于指定核心依赖库的路径, 他们的定义如下:

|  参数名   | 含义  |
|  ----  | ----  |
| CUDA_LIB  | cuda的库路径 |
| CUDNN_LIB | cuDnn的库路径|
| OPENCV_DIR  | OpenCV的安装路径， |
| PADDLE_DIR | Paddle预测库的路径 |

执行下列操作时，**注意**把对应的参数改为你的上述依赖库实际路径：

```shell
cd /root/projects/paddle_models/models/PaddleCV/PaddleDetection/inference

mkdir build && cd build
cmake .. -DWITH_GPU=ON  -DPADDLE_DIR=/root/projects/fluid_inference -DCUDA_LIB=/usr/local/cuda/lib64/ -DOPENCV_DIR=/root/projects/opencv3/ -DCUDNN_LIB=/usr/local/cuda/lib64/
make
```


### Step5: 预测及可视化

执行命令：

```
./detection_demo --conf=/path/to/your/conf --input_dir=/path/to/your/input/data/directory
```

更详细说明请参考ReadMe文档： [预测和可视化部分](../README.md)
