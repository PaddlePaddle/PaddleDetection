[English](README.md) | 简体中文
# PP-PicoDet + PP-TinyPose (Pipeline) CPU-GPU Python部署示例

本目录下提供`det_keypoint_unite_infer.py`快速完成多人模型配置 PP-PicoDet + PP-TinyPose 在CPU/GPU，以及GPU上通过TensorRT加速部署的`单图多人关键点检测`示例。执行如下脚本即可完成.**注意**: PP-TinyPose单模型独立部署，请参考[PP-TinyPose 单模型](../README.md)

## 1. 部署环境准备  
在部署前，需确认软硬件环境，同时下载预编译部署库，参考[FastDeploy安装文档](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install#FastDeploy预编译库安装)安装FastDeploy预编译库。

## 2. 部署模型准备
在部署前，请准备好您所需要运行的推理模型，你可以选择使用[预导出的推理模型](../../README.md)或者[自行导出PaddleDetection部署模型](../../README.md)。  

## 3. 运行部署示例

```bash
# 下载部署示例代码
git clone https://github.com/PaddlePaddle/PaddleDetection.git
cd PaddleDetection/deploy/fastdeploy/cpu-gpu/python/det_keypoint_unite
# 注意：如果当前分支找不到下面的fastdeploy测试代码，请切换到develop分支
# git checkout develop

# 下载PP-TinyPose模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/PP_TinyPose_256x192_infer.tgz
tar -xvf PP_TinyPose_256x192_infer.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/PP_PicoDet_V2_S_Pedestrian_320x320_infer.tgz
tar -xvf PP_PicoDet_V2_S_Pedestrian_320x320_infer.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/000000018491.jpg
# CPU推理
python det_keypoint_unite_infer.py --tinypose_model_dir PP_TinyPose_256x192_infer --det_model_dir PP_PicoDet_V2_S_Pedestrian_320x320_infer --image_file 000000018491.jpg --device cpu
# GPU推理
python det_keypoint_unite_infer.py --tinypose_model_dir PP_TinyPose_256x192_infer --det_model_dir PP_PicoDet_V2_S_Pedestrian_320x320_infer --image_file 000000018491.jpg --device gpu
# GPU上Paddle-TensorRT推理（注意：TensorRT推理第一次运行，有序列化模型的操作，有一定耗时，需要耐心等待）
python det_keypoint_unite_infer.py --tinypose_model_dir PP_TinyPose_256x192_infer --det_model_dir PP_PicoDet_V2_S_Pedestrian_320x320_infer --image_file 000000018491.jpg --device gpu --use_trt True
```

运行完成可视化结果如下图所示
<div  align="center">  
<img src="https://user-images.githubusercontent.com/16222477/196393343-eeb6b68f-0bc6-4927-871f-5ac610da7293.jpeg", width=640px, height=427px />
</div>

- 关于如何通过FastDeploy使用更多不同的推理后端，以及如何使用不同的硬件，请参考文档：[如何切换模型推理后端引擎](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/how_to_change_backend.md)

## 4. 部署示例选项说明  

|参数|含义|默认值
|---|---|---|  
|--tinypose_model_dir|指定关键点模型文件夹所在的路径|None|
|--det_model_dir|指定目标模型文件夹所在的路径|None|
|--image_file|指定测试图片所在的路径|None|  
|--device|指定即将运行的硬件类型，支持的值为`[cpu, gpu]`，当设置为cpu时，可运行在x86 cpu/arm cpu等cpu上|cpu|
|--use_trt|是否使用trt，该项只在device为gpu时有效|False|  

## 5. PPTinyPose 模型串联 Python接口

```python
fd.pipeline.PPTinyPose(det_model=None, pptinypose_model=None)
```

PPTinyPose Pipeline 模型加载和初始化，其中det_model是使用`fd.vision.detection.PicoDet`初始化的检测模型，pptinypose_model是使用`fd.vision.keypointdetection.PPTinyPose`初始化的关键点检测模型。

## 6. 更多指南
- [PaddleDetection Python API文档](https://www.paddlepaddle.org.cn/fastdeploy-api-doc/python/html/object_detection.html)
- [FastDeploy部署PaddleDetection模型概览](../../../)
- [C++部署](../../cpp/)

## 7. 常见问题
- [如何切换模型推理后端引擎](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/how_to_change_backend.md)
- [Intel GPU(独立显卡/集成显卡)的使用](https://github.com/PaddlePaddle/FastDeploy/blob/develop/tutorials/intel_gpu/README.md)
- [编译CPU部署库](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/cpu.md)
- [编译GPU部署库](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/gpu.md)
- [编译Jetson部署库](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/jetson.md)