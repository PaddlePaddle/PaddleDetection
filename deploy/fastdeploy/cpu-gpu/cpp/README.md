[English](README.md) | 简体中文
# PaddleDetection CPU-GPU C++部署示例

本目录下提供`infer.cc`快速完成PPYOLOE模型包括PPYOLOE在CPU/GPU，以及GPU上通过Paddle-TensorRT加速部署的示例。 

## 1. 说明  
PaddleDetection支持利用FastDeploy在NVIDIA GPU、X86 CPU、飞腾CPU、ARM CPU、Intel GPU(独立显卡/集成显卡)硬件上快速部署PaddleDetection模型。FastDeploy目前支持的模型系列，包括但不限于`PPYOLOE`, `PicoDet`, `PaddleYOLOX`, `PPYOLO`, `FasterRCNN`，`SSD`,`PaddleYOLOv5`,`PaddleYOLOv6`,`PaddleYOLOv7`,`RTMDet`,`CascadeRCNN`,`PSSDet`,`RetinaNet`,`PPYOLOESOD`,`FCOS`,`TTFNet`,`TOOD`,`GFL`所有类名的构造函数和预测函数在参数上完全一致。所有模型的调用，只需要参考PPYOLOE的示例，即可快速调用。

## 2. 部署环境准备  
在部署前，需确认软硬件环境，同时下载预编译部署库，参考[FastDeploy安装文档](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install#FastDeploy预编译库安装)安装FastDeploy预编译库。

## 3. 部署模型准备
在部署前，请准备好您所需要运行的推理模型，你可以选择使用[预导出的推理模型](../README.md)或者[自行导出PaddleDetection部署模型](../README.md)。  

## 4. 运行部署示例
以Linux上推理为例，在本目录执行如下命令即可完成编译测试，支持此模型需保证FastDeploy版本1.0.4以上(x.x.x>=1.0.4)

```bash
# 下载FastDeploy预编译库，用户可在上文提到的`FastDeploy预编译库`中自行选择合适的版本使用
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz

# 下载部署示例代码
git clone https://github.com/PaddlePaddle/PaddleDetection.git
cd PaddleDetection/deploy/fastdeploy/cpu-gpu/cpp
# 注意：如果当前分支找不到下面的fastdeploy测试代码，请切换到develop分支
# git checkout develop

# 编译部署示例
mkdir build && cd build
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

# 下载PPYOLOE模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco.tgz
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
tar xvf ppyoloe_crn_l_300e_coco.tgz

# 运行部署示例
# CPU推理
./infer_demo ./ppyoloe_crn_l_300e_coco 000000014439.jpg 0
# GPU推理
./infer_demo ./ppyoloe_crn_l_300e_coco 000000014439.jpg 1
# GPU上GPU上Paddle-TensorRT推理
./infer_demo ./ppyoloe_crn_l_300e_coco 000000014439.jpg 2
```

运行完成可视化结果如下图所示

- 注意，以上命令只适用于Linux或MacOS, Windows下SDK的使用方式请参考: [如何在Windows中使用FastDeploy C++ SDK](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/use_sdk_on_windows.md)  
如果用户使用华为昇腾NPU部署, 请参考以下方式在部署前初始化部署环境:
- 关于如何通过FastDeploy使用更多不同的推理后端，以及如何使用不同的硬件，请参考文档：[如何切换模型推理后端引擎](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/how_to_change_backend.md)  

## 5. 目前已支持的PaddleDetection模型C++类
FastDeploy目前支持的模型系列，包括但不限于`PPYOLOE`, `PicoDet`, `PaddleYOLOX`, `PPYOLO`, `FasterRCNN`，`SSD`,`PaddleYOLOv5`,`PaddleYOLOv6`,`PaddleYOLOv7`,`RTMDet`,`CascadeRCNN`,`PSSDet`,`RetinaNet`,`PPYOLOESOD`,`FCOS`,`TTFNet`,`TOOD`,`GFL`所有类名的构造函数和预测函数在参数上完全一致。所有模型的调用，只需要参考PPYOLOE的示例，即可快速调用。 
```c++
fastdeploy::vision::detection::PicoDet(const string& model_file, const string& params_file, const string& config_file, const RuntimeOption& runtime_option = RuntimeOption(), const ModelFormat& model_format = ModelFormat::PADDLE);
fastdeploy::vision::detection::SOLOv2(const string& model_file, const string& params_file, const string& config_file, const RuntimeOption& runtime_option = RuntimeOption(), const ModelFormat& model_format = ModelFormat::PADDLE);
fastdeploy::vision::detection::PPYOLOE(const string& model_file, const string& params_file, const string& config_file, const RuntimeOption& runtime_option = RuntimeOption(), const ModelFormat& model_format = ModelFormat::PADDLE);
fastdeploy::vision::detection::PPYOLO(const string& model_file, const string& params_file, const string& config_file, const RuntimeOption& runtime_option = RuntimeOption(), const ModelFormat& model_format = ModelFormat::PADDLE);
fastdeploy::vision::detection::YOLOv3(const string& model_file, const string& params_file, const string& config_file, const RuntimeOption& runtime_option = RuntimeOption(), const ModelFormat& model_format = ModelFormat::PADDLE);
fastdeploy::vision::detection::PaddleYOLOX(const string& model_file, const string& params_file, const string& config_file, const RuntimeOption& runtime_option = RuntimeOption(), const ModelFormat& model_format = ModelFormat::PADDLE);
fastdeploy::vision::detection::FasterRCNN(const string& model_file, const string& params_file, const string& config_file, const RuntimeOption& runtime_option = RuntimeOption(), const ModelFormat& model_format = ModelFormat::PADDLE);
fastdeploy::vision::detection::MaskRCNN(const string& model_file, const string& params_file, const string& config_file, const RuntimeOption& runtime_option = RuntimeOption(), const ModelFormat& model_format = ModelFormat::PADDLE);
fastdeploy::vision::detection::SSD(const string& model_file, const string& params_file, const string& config_file, const RuntimeOption& runtime_option = RuntimeOption(), const ModelFormat& model_format = ModelFormat::PADDLE);
fastdeploy::vision::detection::PaddleYOLOv5(const string& model_file, const string& params_file, const string& config_file, const RuntimeOption& runtime_option = RuntimeOption(), const ModelFormat& model_format = ModelFormat::PADDLE);
fastdeploy::vision::detection::PaddleYOLOv6(const string& model_file, const string& params_file, const string& config_file, const RuntimeOption& runtime_option = RuntimeOption(), const ModelFormat& model_format = ModelFormat::PADDLE);
fastdeploy::vision::detection::PaddleYOLOv7(const string& model_file, const string& params_file, const string& config_file, const RuntimeOption& runtime_option = RuntimeOption(), const ModelFormat& model_format = ModelFormat::PADDLE);
fastdeploy::vision::detection::PaddleYOLOv8(const string& model_file, const string& params_file, const string& config_file, const RuntimeOption& runtime_option = RuntimeOption(), const ModelFormat& model_format = ModelFormat::PADDLE);
fastdeploy::vision::detection::CascadeRCNN(const string& model_file, const string& params_file, const string& config_file, const RuntimeOption& runtime_option = RuntimeOption(), const ModelFormat& model_format = ModelFormat::PADDLE);
fastdeploy::vision::detection::PSSDet(const string& model_file, const string& params_file, const string& config_file, const RuntimeOption& runtime_option = RuntimeOption(), const ModelFormat& model_format = ModelFormat::PADDLE);
fastdeploy::vision::detection::RetinaNet(const string& model_file, const string& params_file, const string& config_file, const RuntimeOption& runtime_option = RuntimeOption(), const ModelFormat& model_format = ModelFormat::PADDLE);
fastdeploy::vision::detection::PPYOLOESOD(const string& model_file, const string& params_file, const string& config_file, const RuntimeOption& runtime_option = RuntimeOption(), const ModelFormat& model_format = ModelFormat::PADDLE);
fastdeploy::vision::detection::FCOS(const string& model_file, const string& params_file, const string& config_file, const RuntimeOption& runtime_option = RuntimeOption(), const ModelFormat& model_format = ModelFormat::PADDLE);
fastdeploy::vision::detection::TOOD(const string& model_file, const string& params_file, const string& config_file, const RuntimeOption& runtime_option = RuntimeOption(), const ModelFormat& model_format = ModelFormat::PADDLE);
fastdeploy::vision::detection::GFL(const string& model_file, const string& params_file, const string& config_file, const RuntimeOption& runtime_option = RuntimeOption(), const ModelFormat& model_format = ModelFormat::PADDLE);
```

## 6. 更多指南
- [PaddleDetection C++ API文档](https://www.paddlepaddle.org.cn/fastdeploy-api-doc/cpp/html/namespacefastdeploy_1_1vision_1_1detection.html)
- [FastDeploy部署PaddleDetection模型概览](../../)
- [Python部署](../python)

## 7. 常见问题
- [如何切换模型推理后端引擎](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/how_to_change_backend.md)
- [Intel GPU(独立显卡/集成显卡)的使用](https://github.com/PaddlePaddle/FastDeploy/blob/develop/tutorials/intel_gpu/README.md)
- [编译CPU部署库](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/cpu.md)
- [编译GPU部署库](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/gpu.md)
- [编译Jetson部署库](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/jetson.md)