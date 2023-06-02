# PaddleDetection 算能 C++部署示例

本目录下提供`infer.cc`,`快速完成 PP-YOLOE ,在SOPHGO BM1684x板子上加速部署的示例。PP-YOLOV8和 PicoDet的部署逻辑类似，只需要切换模型即可。

## 1. 部署环境准备
在部署前，需自行编译基于算能硬件的预测库，参考文档[算能硬件部署环境](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install#算能硬件部署环境)

## 2. 部署模型准备  
在部署前，请准备好您所需要运行的推理模型，你可以选择使用[预导出的推理模型](../README.md)或者[自行导出PaddleDetection部署模型](../README.md)。

## 3. 生成基本目录文件

该例程由以下几个部分组成
```text
.
├── CMakeLists.txt
├── fastdeploy-sophgo  # 编译文件夹
├── image  # 存放图片的文件夹
├── infer.cc
└── model  # 存放模型文件的文件夹
```

## 4. 运行部署示例

### 4.1 编译并拷贝SDK到thirdpartys文件夹

请参考[SOPHGO部署库编译](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/sophgo.md)仓库编译SDK，编译完成后，将在build目录下生成fastdeploy-sophgo目录.

### 4.2 拷贝模型文件，以及配置文件至model文件夹
将Paddle模型转换为SOPHGO bmodel模型，转换步骤参考[文档](../README.md)  
将转换后的SOPHGO bmodel模型文件拷贝至model中

### 4.3 准备测试图片至image文件夹
```bash
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
cp 000000014439.jpg ./images
```

### 4.4 编译example

```bash
cd build
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-sophgo
make
```

## 4.5 运行例程

```bash
#ppyoloe推理示例
./infer_demo model images/000000014439.jpg
```

## 5. 更多指南
- [FastDeploy部署PaddleDetection模型概览](../../)
- [Python部署](../python)
- [模型转换](../README.md)