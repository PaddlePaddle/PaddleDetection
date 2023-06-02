[English](README.md) | 简体中文
# PaddleDetection CPU-GPU Python部署示例

本目录下提供`infer.py`快速完成PPYOLOE模型包括PPYOLOE在CPU/GPU，以及GPU上通过Paddle-TensorRT加速部署的示例。 

## 1. 说明  
PaddleDetection支持利用FastDeploy在NVIDIA GPU、X86 CPU、飞腾CPU、ARM CPU、Intel GPU(独立显卡/集成显卡)硬件上快速部署PaddleDetection模型。FastDeploy目前支持的模型系列，包括但不限于`PPYOLOE`, `PicoDet`, `PaddleYOLOX`, `PPYOLO`, `FasterRCNN`，`SSD`,`PaddleYOLOv5`,`PaddleYOLOv6`,`PaddleYOLOv7`,`RTMDet`,`CascadeRCNN`,`PSSDet`,`RetinaNet`,`PPYOLOESOD`,`FCOS`,`TTFNet`,`TOOD`,`GFL`所有类名的构造函数和预测函数在参数上完全一致。所有模型的调用，只需要参考PPYOLOE的示例，即可快速调用。

## 2. 部署环境准备  
在部署前，需确认软硬件环境，同时下载预编译部署库，参考[FastDeploy安装文档](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install#FastDeploy预编译库安装)安装FastDeploy预编译库。

## 3. 部署模型准备
在部署前，请准备好您所需要运行的推理模型，你可以选择使用[预导出的推理模型](../README.md)或者[自行导出PaddleDetection部署模型](../README.md)。    

## 4. 运行部署示例
以Linux上推理为例，在本目录执行如下命令即可完成编译测试，支持此模型需保证FastDeploy版本1.0.4以上(x.x.x>=1.0.4)

### 4.1 目标检测示例
```bash
# 下载部署示例代码
git clone https://github.com/PaddlePaddle/PaddleDetection.git
cd PaddleDetection/deploy/fastdeploy/cpu-gpu/python
# 注意：如果当前分支找不到下面的fastdeploy测试代码，请切换到develop分支
# git checkout develop

# 下载PPYOLOE模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco.tgz
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
tar xvf ppyoloe_crn_l_300e_coco.tgz

# 运行部署示例
# CPU推理
python infer.py --model_dir ppyoloe_crn_l_300e_coco --image_file 000000014439.jpg --device cpu
# GPU推理
python infer.py --model_dir ppyoloe_crn_l_300e_coco --image_file 000000014439.jpg --device gpu
# GPU上Paddle-TensorRT推理 （注意：TensorRT推理第一次运行，有序列化模型的操作，有一定耗时，需要耐心等待）
python infer.py --model_dir ppyoloe_crn_l_300e_coco --image_file 000000014439.jpg --device gpu --use_trt True
```

运行完成可视化结果如下图所示
<div  align="center">  
<img src="https://user-images.githubusercontent.com/19339784/184326520-7075e907-10ed-4fad-93f8-52d0e35d4964.jpg", width=480px, height=320px />
</div>

### 4.2 关键点检测示例  
```bash
# 下载部署示例代码
git clone https://github.com/PaddlePaddle/PaddleDetection.git
cd PaddleDetection/deploy/fastdeploy/cpu-gpu/python
# 注意：如果当前分支找不到下面的fastdeploy测试代码，请切换到develop分支
# git checkout develop

# 下载PP-TinyPose模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/PP_TinyPose_256x192_infer.tgz
tar -xvf PP_TinyPose_256x192_infer.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/hrnet_demo.jpg

# 运行部署示例
# CPU推理
python pptinypose_infer.py --model_dir PP_TinyPose_256x192_infer --image_file hrnet_demo.jpg --device cpu
# GPU推理
python pptinypose_infer.py --model_dir PP_TinyPose_256x192_infer --image_file hrnet_demo.jpg --device gpu
# GPU上Paddle-TensorRT推理 （注意：TensorRT推理第一次运行，有序列化模型的操作，有一定耗时，需要耐心等待）
python pptinypose_infer.py --model_dir PP_TinyPose_256x192_infer --image_file hrnet_demo.jpg --device gpu --use_trt True
```

运行完成可视化结果如下图所示
<div  align="center">  
<img src="https://user-images.githubusercontent.com/16222477/196386764-dd51ad56-c410-4c54-9580-643f282f5a83.jpeg", width=359px, height=423px />
</div>  

关于如何进行多人关键点检测，请参考[PPTinyPose Pipeline示例](./det_keypoint_unite/)

## 5. 部署示例选项说明  

|参数|含义|默认值
|---|---|---|  
|--model_dir|指定模型文件夹所在的路径|None|
|--image_file|指定测试图片所在的路径|None|  
|--device|指定即将运行的硬件类型，支持的值为`[cpu, gpu]`，当设置为cpu时，可运行在x86 cpu/arm cpu等cpu上|cpu|
|--use_trt|是否使用trt，该项只在device为gpu时有效|False|  

## 6. PaddleDetection Python接口
FastDeploy目前支持的模型系列，包括但不限于`PPYOLOE`, `PicoDet`, `PaddleYOLOX`, `PPYOLO`, `FasterRCNN`，`SSD`,`PaddleYOLOv5`,`PaddleYOLOv6`,`PaddleYOLOv7`,`RTMDet`,`CascadeRCNN`,`PSSDet`,`RetinaNet`,`PPYOLOESOD`,`FCOS`,`TTFNet`,`TOOD`,`GFL`所有类名的构造函数和预测函数在参数上完全一致。所有模型的调用，只需要参考PPYOLOE的示例，即可快速调用。 

### 6.1 目标检测及实例分割模型
```python
fastdeploy.vision.detection.PPYOLOE(model_file, params_file, config_file, runtime_option=None, model_format=ModelFormat.PADDLE)
fastdeploy.vision.detection.PicoDet(model_file, params_file, config_file, runtime_option=None, model_format=ModelFormat.PADDLE)
fastdeploy.vision.detection.PaddleYOLOX(model_file, params_file, config_file, runtime_option=None, model_format=ModelFormat.PADDLE)
fastdeploy.vision.detection.YOLOv3(model_file, params_file, config_file, runtime_option=None, model_format=ModelFormat.PADDLE)
fastdeploy.vision.detection.PPYOLO(model_file, params_file, config_file, runtime_option=None, model_format=ModelFormat.PADDLE)
fastdeploy.vision.detection.FasterRCNN(model_file, params_file, config_file, runtime_option=None, model_format=ModelFormat.PADDLE)
fastdeploy.vision.detection.MaskRCNN(model_file, params_file, config_file, runtime_option=None, model_format=ModelFormat.PADDLE)
fastdeploy.vision.detection.SSD(model_file, params_file, config_file, runtime_option=None, model_format=ModelFormat.PADDLE)
fastdeploy.vision.detection.PaddleYOLOv5(model_file, params_file, config_file, runtime_option=None, model_format=ModelFormat.PADDLE)
fastdeploy.vision.detection.PaddleYOLOv6(model_file, params_file, config_file, runtime_option=None, model_format=ModelFormat.PADDLE)
fastdeploy.vision.detection.PaddleYOLOv7(model_file, params_file, config_file, runtime_option=None, model_format=ModelFormat.PADDLE)
fastdeploy.vision.detection.RTMDet(model_file, params_file, config_file, runtime_option=None, model_format=ModelFormat.PADDLE)
fastdeploy.vision.detection.CascadeRCNN(model_file, params_file, config_file, runtime_option=None, model_format=ModelFormat.PADDLE)
fastdeploy.vision.detection.PSSDet(model_file, params_file, config_file, runtime_option=None, model_format=ModelFormat.PADDLE)
fastdeploy.vision.detection.RetinaNet(model_file, params_file, config_file, runtime_option=None, model_format=ModelFormat.PADDLE)
fastdeploy.vision.detection.PPYOLOESOD(model_file, params_file, config_file, runtime_option=None, model_format=ModelFormat.PADDLE)
fastdeploy.vision.detection.FCOS(model_file, params_file, config_file, runtime_option=None, model_format=ModelFormat.PADDLE)
fastdeploy.vision.detection.TTFNet(model_file, params_file, config_file, runtime_option=None, model_format=ModelFormat.PADDLE)
fastdeploy.vision.detection.TOOD(model_file, params_file, config_file, runtime_option=None, model_format=ModelFormat.PADDLE)
fastdeploy.vision.detection.GFL(model_file, params_file, config_file, runtime_option=None, model_format=ModelFormat.PADDLE)
```
### 6.2 关键点检测模型  
```python
fd.vision.keypointdetection.PPTinyPose(model_file, params_file, config_file, runtime_option=None, model_format=ModelFormat.PADDLE)
```

PaddleDetection模型加载和初始化，其中model_file， params_file为导出的Paddle部署模型格式, config_file为PaddleDetection同时导出的部署配置yaml文件

## 7. 更多指南
- [PaddleDetection Python API文档](https://www.paddlepaddle.org.cn/fastdeploy-api-doc/python/html/object_detection.html)
- [FastDeploy部署PaddleDetection模型概览](../../)
- [C++部署](../cpp)

## 8. 常见问题
- [如何切换模型推理后端引擎](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/how_to_change_backend.md)
- [Intel GPU(独立显卡/集成显卡)的使用](https://github.com/PaddlePaddle/FastDeploy/blob/develop/tutorials/intel_gpu/README.md)
- [编译CPU部署库](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/cpu.md)
- [编译GPU部署库](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/gpu.md)
- [编译Jetson部署库](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/jetson.md)