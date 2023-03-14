[English](README.md) | 简体中文
# PaddleDetection C++部署示例

本目录下提供`infer_xxx.cc`快速完成PaddleDetection模型包括PPYOLOE/PicoDet/YOLOX/YOLOv3/PPYOLO/FasterRCNN/YOLOv5/YOLOv6/YOLOv7/RTMDet/CascadeRCNN/PSSDet/RetinaNet/PPYOLOESOD/FCOS/TTFNet/TOOD/GFL在CPU/GPU，以及GPU上通过TensorRT加速部署的示例。

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. 根据开发环境，下载预编译部署库和samples代码，参考[FastDeploy预编译库](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

以Linux上推理为例，在本目录执行如下命令即可完成编译测试，支持此模型需保证FastDeploy版本0.7.0以上(x.x.x>=0.7.0)

```bash
以ppyoloe为例进行推理部署

mkdir build
cd build
# 下载FastDeploy预编译库，用户可在上文提到的`FastDeploy预编译库`中自行选择合适的版本使用
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

# 下载PPYOLOE模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco.tgz
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
tar xvf ppyoloe_crn_l_300e_coco.tgz


# CPU推理
./infer_ppyoloe_demo ./ppyoloe_crn_l_300e_coco 000000014439.jpg 0
# GPU推理
./infer_ppyoloe_demo ./ppyoloe_crn_l_300e_coco 000000014439.jpg 1
# GPU上TensorRT推理
./infer_ppyoloe_demo ./ppyoloe_crn_l_300e_coco 000000014439.jpg 2
# 昆仑芯XPU推理
./infer_ppyoloe_demo ./ppyoloe_crn_l_300e_coco 000000014439.jpg 3
# 华为昇腾推理
./infer_ppyoloe_demo ./ppyoloe_crn_l_300e_coco 000000014439.jpg 4
```

以上命令只适用于Linux或MacOS, Windows下SDK的使用方式请参考:  
- [如何在Windows中使用FastDeploy C++ SDK](../../../../../docs/cn/faq/use_sdk_on_windows.md)

如果用户使用华为昇腾NPU部署, 请参考以下方式在部署前初始化部署环境:
- [如何使用华为昇腾NPU部署](../../../../../docs/cn/faq/use_sdk_on_ascend.md)

## PaddleDetection C++接口

### 模型类

PaddleDetection目前支持6种模型系列，类名分别为`PPYOLOE`, `PicoDet`, `PaddleYOLOX`, `PPYOLO`, `FasterRCNN`，`SSD`,`PaddleYOLOv5`,`PaddleYOLOv6`,`PaddleYOLOv7`,`RTMDet`,`CascadeRCNN`,`PSSDet`,`RetinaNet`,`PPYOLOESOD`,`FCOS`,`TTFNet`,`TOOD`,`GFL`所有类名的构造函数和预测函数在参数上完全一致，本文档以PPYOLOE为例讲解API
```c++
fastdeploy::vision::detection::PPYOLOE(
        const string& model_file,
        const string& params_file,
        const string& config_file
        const RuntimeOption& runtime_option = RuntimeOption(),
        const ModelFormat& model_format = ModelFormat::PADDLE)
```

PaddleDetection PPYOLOE模型加载和初始化，其中model_file为导出的ONNX模型格式。

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径
> * **config_file**(str): 配置文件路径，即PaddleDetection导出的部署yaml文件
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(ModelFormat): 模型格式，默认为PADDLE格式

#### Predict函数

> ```c++
> PPYOLOE::Predict(cv::Mat* im, DetectionResult* result)
> ```
>
> 模型预测接口，输入图像直接输出检测结果。
>
> **参数**
>
> > * **im**: 输入图像，注意需为HWC，BGR格式
> > * **result**: 检测结果，包括检测框，各个框的置信度, DetectionResult说明参考[视觉模型预测结果](../../../../../docs/api/vision_results/)

- [模型介绍](../../)
- [Python部署](../python)
- [视觉模型预测结果](../../../../../docs/api/vision_results/)
- [如何切换模型推理后端引擎](../../../../../docs/cn/faq/how_to_change_backend.md)
