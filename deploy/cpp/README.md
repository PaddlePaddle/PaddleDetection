# C++端预测部署



## 各环境编译部署教程
- [Linux 编译部署](docs/linux_build.md)
- [Windows编译部署(使用Visual Studio 2019)](docs/windows_vs2019_build.md)
- [NV Jetson编译部署](docs/Jetson_build.md)


## C++部署总览
[1.说明](#1说明)

[2.主要目录和文件](#2主要目录和文件)


### 1.说明

本目录为用户提供一个跨平台的`C++`部署方案，让用户通过`PaddleDetection`训练的模型导出后，即可基于本项目快速运行，也可以快速集成代码结合到自己的项目实际应用中去。

主要设计的目标包括以下四点：
- 跨平台，支持在 `Windows` 和 `Linux` 完成编译、二次开发集成和部署运行
- 可扩展性，支持用户针对新模型开发自己特殊的数据预处理等逻辑
- 高性能，除了`PaddlePaddle`自身带来的性能优势，我们还针对图像检测的特点对关键步骤进行了性能优化
- 支持各种不同检测模型结构，包括`Yolov3`/`Faster_RCNN`/`SSD`等

### 2.主要目录和文件

```bash
deploy/cpp
|
├── src
│   ├── main.cc # 集成代码示例, 程序入口
│   ├── object_detector.cc # 模型加载和预测主要逻辑封装类实现
│   └── preprocess_op.cc # 预处理相关主要逻辑封装实现
|
├── include
│   ├── config_parser.h # 导出模型配置yaml文件解析
│   ├── object_detector.h # 模型加载和预测主要逻辑封装类
│   └── preprocess_op.h # 预处理相关主要逻辑类封装
|
├── docs
│   ├── linux_build.md # Linux 编译指南
│   └── windows_vs2019_build.md # Windows VS2019编译指南
│
├── build.sh # 编译命令脚本
│
├── CMakeList.txt # cmake编译入口文件
|
├── CMakeSettings.json # Visual Studio 2019 CMake项目编译设置
│
└── cmake # 依赖的外部项目cmake（目前仅有yaml-cpp）

```
