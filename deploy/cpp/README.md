# C++端预测部署

## 本教程结构

[1.说明](#1说明)

[2.主要目录和文件](#2主要目录和文件)

[3.编译部署](#3编译)



## 1.说明

本目录为用户提供一个跨平台的`C++`部署方案，让用户通过`PaddleDetection`训练的模型导出后，即可基于本项目快速运行，也可以快速集成代码结合到自己的项目实际应用中去。

主要设计的目标包括以下四点：
- 跨平台，支持在 `Windows` 和 `Linux` 完成编译、二次开发集成和部署运行
- 可扩展性，支持用户针对新模型开发自己特殊的数据预处理等逻辑
- 高性能，除了`PaddlePaddle`自身带来的性能优势，我们还针对图像检测的特点对关键步骤进行了性能优化
- 支持各种不同检测模型结构，包括`Yolov3`/`Faster_RCNN`/`SSD`/`RetinaNet`等

## 2.主要目录和文件

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

## 3.编译部署

### 3.1 导出模型
请确认您已经基于`PaddleDetection`的[export_model.py](https://github.com/PaddlePaddle/PaddleDetection/blob/master/tools/export_model.py)导出您的模型，并妥善保存到合适的位置。导出模型细节请参考 [导出模型教程](https://github.com/PaddlePaddle/PaddleDetection/blob/master/docs/advanced_tutorials/deploy/EXPORT_MODEL.md)。

模型导出后, 目录结构如下(以`yolov3_darknet`为例):
```
yolov3_darknet # 模型目录
├── infer_cfg.yml # 模型配置信息
├── __model__     # 模型文件
└── __params__    # 参数文件
```

预测时，该目录所在的路径会作为程序的输入参数。

### 3.2 编译

仅支持在`Windows`和`Linux`平台编译和使用
- [Linux 编译指南](docs/linux_build.md)
- [Windows编译指南(使用Visual Studio 2019)](docs/windows_vs2019_build.md)
