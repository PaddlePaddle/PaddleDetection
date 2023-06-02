[English](README.md) | 简体中文
# PP-PicoDet + PP-TinyPose (Pipeline) CPU-GPU C++部署示例

本目录下提供`det_keypoint_unite_infer.cc`快速完成多人模型配置 PP-PicoDet + PP-TinyPose 在CPU/GPU，以及GPU上通过TensorRT加速部署的`单图多人关键点检测`示例。执行如下脚本即可完成。**注意**: PP-TinyPose单模型独立部署，请参考[PP-TinyPose 单模型](../README.md)

## 1. 部署环境准备  
在部署前，需确认软硬件环境，同时下载预编译部署库，参考[FastDeploy安装文档](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install#FastDeploy预编译库安装)安装FastDeploy预编译库。

## 2. 部署模型准备
在部署前，请准备好您所需要运行的推理模型，你可以选择使用[预导出的推理模型](../../README.md)或者[自行导出PaddleDetection部署模型](../../README.md)。  

## 3. 运行部署示例
以Linux上推理为例，在本目录执行如下命令即可完成编译测试，支持此模型需保证FastDeploy版本1.0.4以上(x.x.x>=1.0.4)

```bash
mkdir build
cd build
# 下载FastDeploy预编译库，用户可在上文提到的`FastDeploy预编译库`中自行选择合适的版本使用
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

# 下载部署示例代码
git clone https://github.com/PaddlePaddle/PaddleDetection.git
cd PaddleDetection/deploy/fastdeploy/cpu-gpu/cpp/det_keypoint_unite
# 注意：如果当前分支找不到下面的fastdeploy测试代码，请切换到develop分支
# git checkout develop

# 下载PP-TinyPose和PP-PicoDet模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/PP_TinyPose_256x192_infer.tgz
tar -xvf PP_TinyPose_256x192_infer.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/PP_PicoDet_V2_S_Pedestrian_320x320_infer.tgz
tar -xvf PP_PicoDet_V2_S_Pedestrian_320x320_infer.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/000000018491.jpg

# CPU推理
./infer_demo PP_PicoDet_V2_S_Pedestrian_320x320_infer PP_TinyPose_256x192_infer 000000018491.jpg 0
# GPU推理
./infer_demo PP_PicoDet_V2_S_Pedestrian_320x320_infer PP_TinyPose_256x192_infer 000000018491.jpg 1
# GPU上Paddle-TensorRT推理（注意：TensorRT推理第一次运行，有序列化模型的操作，有一定耗时，需要耐心等待）
./infer_demo PP_PicoDet_V2_S_Pedestrian_320x320_infer PP_TinyPose_256x192_infer 000000018491.jpg 2
```

运行完成可视化结果如下图所示
<div  align="center">  
<img src="https://user-images.githubusercontent.com/16222477/196393343-eeb6b68f-0bc6-4927-871f-5ac610da7293.jpeg", width=359px, height=423px />
</div>  

- 注意，以上命令只适用于Linux或MacOS, Windows下SDK的使用方式请参考: [如何在Windows中使用FastDeploy C++ SDK](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/use_sdk_on_windows.md)  
- 关于如何通过FastDeploy使用更多不同的推理后端，以及如何使用不同的硬件，请参考文档：[如何切换模型推理后端引擎](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/how_to_change_backend.md) 

## 4. PP-TinyPose 模型串联 C++ 接口

```c++
fastdeploy::pipeline::PPTinyPose(
        fastdeploy::vision::detection::PicoDet* det_model,
        fastdeploy::vision::keypointdetection::PPTinyPose* pptinypose_model)
```

PPTinyPose Pipeline模型加载和初始化。det_model表示初始化后的检测模型，pptinypose_model表示初始化后的关键点检测模型。


## 5. 更多指南
- [PaddleDetection C++ API文档](https://www.paddlepaddle.org.cn/fastdeploy-api-doc/cpp/html/namespacefastdeploy_1_1vision_1_1detection.html)
- [FastDeploy部署PaddleDetection模型概览](../../../)
- [Python部署](../../python/det_keypoint_unite/)

## 6. 常见问题
- [如何切换模型推理后端引擎](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/how_to_change_backend.md)
- [Intel GPU(独立显卡/集成显卡)的使用](https://github.com/PaddlePaddle/FastDeploy/blob/develop/tutorials/intel_gpu/README.md)
- [编译CPU部署库](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/cpu.md)
- [编译GPU部署库](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/gpu.md)
- [编译Jetson部署库](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/jetson.md)