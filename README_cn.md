简体中文 | [English](README_en.md)

<div align="center">
<p align="center">
  <img src="https://user-images.githubusercontent.com/48054808/160532560-34cf7a1f-d950-435e-90d2-4b0a679e5119.png" align="middle" width = "800" />
</p>

**飞桨目标检测开发套件，端到端地完成从训练到部署的全流程目标检测应用。**

<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href="https://github.com/PaddlePaddle/PaddleDetection/releases"><img src="https://img.shields.io/github/v/release/PaddlePaddle/PaddleDetection?color=ffa"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href="https://github.com/PaddlePaddle/PaddleDetection/stargazers"><img src="https://img.shields.io/github/stars/PaddlePaddle/PaddleDetection?color=ccf"></a>
</p>
</div>

## <img src="https://user-images.githubusercontent.com/48054808/157793354-6e7f381a-0aa6-4bb7-845c-9acf2ecc05c3.png" width="20"/> 产品动态

- 🔥 **2022.3.24：PaddleDetection发布[release/2.4版本](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4)**

  - 发布高精度云边一体SOTA目标检测模型[PP-YOLOE](configs/ppyoloe)，发布s/m/l/x版本，l版本COCO test2017数据集精度51.4%，V100预测速度78.1 FPS，支持混合精度训练，训练较PP-YOLOv2加速33%，全系列多尺度模型，满足不同硬件算力需求，可适配服务器、边缘端GPU及其他服务器端AI加速卡。
  - 发布边缘端和CPU端超轻量SOTA目标检测模型[PP-PicoDet增强版](configs/picodet)，精度提升2%左右，CPU预测速度提升63%，新增参数量0.7M的PicoDet-XS模型，提供模型稀疏化和量化功能，便于模型加速，各类硬件无需单独开发后处理模块，降低部署门槛。
  - 发布实时行人分析工具[PP-Human](deploy/pphuman)，支持行人跟踪、人流量统计、人体属性识别与摔倒检测四大能力，基于真实场景数据特殊优化，精准识别各类摔倒姿势，适应不同环境背景、光线及摄像角度。
  - 新增[YOLOX](configs/yolox)目标检测模型，支持nano/tiny/s/m/l/x版本，x版本COCO val2017数据集精度51.8%。

- 2021.11.03: PaddleDetection发布[release/2.3版本](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3)

  - 发布轻量级检测特色模型⚡[PP-PicoDet](configs/picodet)，0.99m的参数量可实现精度30+mAP、速度150FPS。
  - 发布轻量级关键点特色模型⚡[PP-TinyPose](configs/keypoint/tiny_pose)，单人场景FP16推理可达122FPS、51.8AP，具有精度高速度快、检测人数无限制、微小目标效果好的优势。
  - 发布实时跟踪系统[PP-Tracking](deploy/pptracking)，覆盖单、多镜头下行人、车辆、多类别跟踪，对小目标、密集型特殊优化，提供人、车流量技术解决方案。
  - 新增[Swin Transformer](configs/faster_rcnn)，[TOOD](configs/tood)，[GFL](configs/gfl)目标检测模型。
  - 发布[Sniper](configs/sniper)小目标检测优化模型，发布针对EdgeBoard优化[PP-YOLO-EB](configs/ppyolo)模型。
  - 新增轻量化关键点模型[Lite HRNet](configs/keypoint)关键点模型并支持Paddle Lite部署。

- 2021.08.10: PaddleDetection发布[release/2.2版本](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.2)

  - 发布Transformer检测系列模型，包括[DETR](configs/detr), [Deformable DETR](configs/deformable_detr), [Sparse RCNN](configs/sparse_rcnn)。
  - 新增Dark HRNet关键点模型和MPII数据集[关键点模型](configs/keypoint)
  - 新增[人头](configs/mot/headtracking21)、[车辆](configs/mot/vehicle)跟踪垂类模型。

- 2021.05.20: PaddleDetection发布[release/2.1版本](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.1)

  - 新增[关键点检测](configs/keypoint)，模型包括HigherHRNet，HRNet。
  - 新增[多目标跟踪](configs/mot)能力，模型包括DeepSORT，JDE，FairMOT。
  - 发布PPYOLO系列模型压缩模型，新增[ONNX模型导出教程](deploy/EXPORT_ONNX_MODEL.md)。

## <img title="" src="https://user-images.githubusercontent.com/48054808/157795569-9fc77c85-732f-4870-9be0-99a7fe2cff27.png" alt="" width="20"> 简介

**PaddleDetection**为基于飞桨PaddlePaddle的端到端目标检测套件，内置**30+模型算法**及**250+预训练模型**，覆盖**目标检测、实例分割、跟踪、关键点检测**等方向，其中包括**服务器端和移动端高精度、轻量级**产业级SOTA模型、冠军方案和学术前沿算法，并提供配置化的网络模块组件、十余种数据增强策略和损失函数等高阶优化支持和多种部署方案，在打通数据处理、模型开发、训练、压缩、部署全流程的基础上，提供丰富的案例及教程，加速算法产业落地应用。

#### 提供目标检测、实例分割、多目标跟踪、关键点检测等多种能力

<div  align="center">
  <img src="docs/images/ppdet.gif" width="800"/>
</div>

#### 应用场景覆盖工业、智慧城市、安防、交通、零售、医疗等十余种行业

<div  align="center">
  <img src="https://user-images.githubusercontent.com/48054808/157826886-2e101a71-25a2-42f5-bf5e-30a97be28f46.gif" width="800"/>
</div>

## <img src="https://user-images.githubusercontent.com/48054808/157799599-e6a66855-bac6-4e75-b9c0-96e13cb9612f.png" width="20"/> 特性

- **模型丰富**: 包含**目标检测**、**实例分割**、**人脸检测**、****关键点检测****、**多目标跟踪**等**250+个预训练模型**，涵盖多种**全球竞赛冠军**方案。
- **使用简洁**：模块化设计，解耦各个网络组件，开发者轻松搭建、试用各种检测模型及优化策略，快速得到高性能、定制化的算法。
- **端到端打通**: 从数据增强、组网、训练、压缩、部署端到端打通，并完备支持**云端**/**边缘端**多架构、多设备部署。
- **高性能**: 基于飞桨的高性能内核，模型训练速度及显存占用优势明显。支持FP16训练, 支持多机训练。

## <img title="" src="https://user-images.githubusercontent.com/48054808/157800467-2a9946ad-30d1-49a9-b9db-ba33413d9c90.png" alt="" width="20"> 技术交流

- 如果你发现任何PaddleDetection存在的问题或者是建议, 欢迎通过[GitHub Issues](https://github.com/PaddlePaddle/PaddleDetection/issues)给我们提issues。

- 欢迎加入PaddleDetection QQ、微信用户群（添加并回复小助手“检测”）

  <div align="center">
  <img src="https://user-images.githubusercontent.com/48054808/157800129-2f9a0b72-6bb8-4b10-8310-93ab1639253f.jpg"  width = "200" />  
  <img src="https://user-images.githubusercontent.com/48054808/160531099-9811bbe6-cfbb-47d5-8bdb-c2b40684d7dd.png"  width = "200" />  
  </div>

## <img src="https://user-images.githubusercontent.com/48054808/157827140-03ffaff7-7d14-48b4-9440-c38986ea378c.png" width="20"/> 套件结构概览

<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Architectures</b>
      </td>
      <td>
        <b>Backbones</b>
      </td>
      <td>
        <b>Components</b>
      </td>
      <td>
        <b>Data Augmentation</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
          <li><b>Object Detection</b></li>
          <ul>
            <li>Faster RCNN</li>
            <li>FPN</li>
            <li>Cascade-RCNN</li>
            <li>PSS-Det</li>
            <li>RetinaNet</li>
            <li>YOLOv3</li>  
            <li>PP-YOLOv1/v2</li>
            <li>PP-YOLO-Tiny</li>
            <li>PP-YOLOE</li>
            <li>YOLOX</li>
            <li>SSD</li>
            <li>CenterNet</li>
            <li>FCOS</li>  
            <li>TTFNet</li>
            <li>TOOD</li>
            <li>GFL</li>
            <li>PP-PicoDet</li>
            <li>DETR</li>
            <li>Deformable DETR</li>
            <li>Swin Transformer</li>
            <li>Sparse RCNN</li>
        </ul>
        <li><b>Instance Segmentation</b></li>
        <ul>
            <li>Mask RCNN</li>
            <li>SOLOv2</li>
        </ul>
        <li><b>Face Detection</b></li>
        <ul>
            <li>BlazeFace</li>
        </ul>
        <li><b>Multi-Object-Tracking</b></li>
        <ul>
            <li>JDE</li>
            <li>FairMOT</li>
            <li>DeepSORT</li>
            <li>ByteTrack</li>
        </ul>
        <li><b>KeyPoint-Detection</b></li>
        <ul>
            <li>HRNet</li>
            <li>HigherHRNet</li>
            <li>Lite-HRNet</li>
            <li>PP-TinyPose</li>
        </ul>
      </ul>
      </td>
      <td>
        <ul>
          <li>ResNet(&vd)</li>
          <li>Res2Net(&vd)</li>
          <li>CSPResNet</li>
          <li>SENet</li>
          <li>Res2Net</li>
          <li>HRNet</li>
          <li>Lite-HRNet</li>
          <li>DarkNet</li>
          <li>CSPDarkNet</li>
          <li>MobileNetv1/v3</li>  
          <li>ShuffleNet</li>
          <li>GhostNet</li>
          <li>BlazeNet</li>
          <li>DLA</li>
          <li>HardNet</li>
          <li>LCNet</li>  
          <li>ESNet</li>  
          <li>Swin-Transformer</li>
        </ul>
      </td>
      <td>
        <ul><li><b>Common</b></li>
          <ul>
            <li>Sync-BN</li>
            <li>Group Norm</li>
            <li>DCNv2</li>
            <li>EMA</li>
          </ul>  
        </ul>
        <ul><li><b>KeyPoint</b></li>
          <ul>
            <li>DarkPose</li>
          </ul>  
        </ul>
        <ul><li><b>FPN</b></li>
          <ul>
            <li>BiFPN</li>
            <li>CSP-PAN</li>
            <li>Custom-PAN</li>
            <li>ES-PAN</li>
            <li>HRFPN</li>
          </ul>  
        </ul>  
        <ul><li><b>Loss</b></li>
          <ul>
            <li>Smooth-L1</li>
            <li>GIoU/DIoU/CIoU</li>  
            <li>IoUAware</li>
            <li>Focal Loss</li>
            <li>CT Focal Loss</li>
            <li>VariFocal Loss</li>
          </ul>  
        </ul>  
        <ul><li><b>Post-processing</b></li>
          <ul>
            <li>SoftNMS</li>
            <li>MatrixNMS</li>  
          </ul>  
        </ul>
        <ul><li><b>Speed</b></li>
          <ul>
            <li>FP16 training</li>
            <li>Multi-machine training </li>  
          </ul>  
        </ul>  
      </td>
      <td>
        <ul>
          <li>Resize</li>  
          <li>Lighting</li>  
          <li>Flipping</li>  
          <li>Expand</li>
          <li>Crop</li>
          <li>Color Distort</li>  
          <li>Random Erasing</li>  
          <li>Mixup </li>
          <li>AugmentHSV</li>
          <li>Mosaic</li>
          <li>Cutmix </li>
          <li>Grid Mask</li>
          <li>Auto Augment</li>  
          <li>Random Perspective</li>  
        </ul>  
      </td>  
    </tr>

</td>
    </tr>
  </tbody>
</table>

## <img src="https://user-images.githubusercontent.com/48054808/157801371-9a9a8c65-1690-4123-985a-e0559a7f9494.png" width="20"/> 模型性能概览

各模型结构和骨干网络的代表模型在COCO数据集上精度mAP和单卡Tesla V100上预测速度(FPS)对比图。

<div align="center">
  <img src="docs/images/fps_map.png" />
</div>

**说明：**

- `CBResNet`为`Cascade-Faster-RCNN-CBResNet200vd-FPN`模型，COCO数据集mAP高达53.3%
- `Cascade-Faster-RCNN`为`Cascade-Faster-RCNN-ResNet50vd-DCN`，PaddleDetection将其优化到COCO数据mAP为47.8%时推理速度为20FPS
- `PP-YOLO`在COCO数据集精度45.9%，Tesla V100预测速度72.9FPS，精度速度均优于[YOLOv4](https://arxiv.org/abs/2004.10934)
- `PP-YOLO v2`是对`PP-YOLO`模型的进一步优化，在COCO数据集精度49.5%，Tesla V100预测速度68.9FPS
- `PP-YOLOE`是对`PP-YOLO v2`模型的进一步优化，在COCO数据集精度51.4%，Tesla V100预测速度78.1FPS
- 图中模型均可在[模型库](#模型库)中获取

各移动端模型在COCO数据集上精度mAP和高通骁龙865处理器上预测速度(FPS)对比图。

<div align="center">
  <img src="docs/images/mobile_fps_map.png" width=600/>
</div>

**说明：**

- 测试数据均使用高通骁龙865(4\*A77 + 4\*A55)处理器batch size为1, 开启4线程测试，测试使用NCNN预测库，测试脚本见[MobileDetBenchmark](https://github.com/JiweiMaster/MobileDetBenchmark)
- [PP-PicoDet](configs/picodet)及[PP-YOLO-Tiny](configs/ppyolo)为PaddleDetection自研模型，其余模型PaddleDetection暂未提供

## <img src="https://user-images.githubusercontent.com/48054808/157829890-a535b8a6-631c-4c87-b861-64d4b32b2d6a.png" width="20"/> 模型库

### 1. 通用检测

#### PP-YOLOE系列 推荐场景：Nvidia V100, T4等云端GPU和Jetson系列等边缘端设备

| 模型名称   |   COCO精度（mAP）| V100 TensorRT FP16速度(FPS) |  配置文件  |   模型下载  |
| :-------- | :---------------: | :------------: |:-------: |:-------: |
|  PP-YOLOE-s |     42.7      |    333.3    |    [链接](configs/ppyoloe/ppyoloe_crn_l_300e_coco.yml) | [下载地址](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams) |
| PP-YOLOE-m |     48.6      |    208.3    |    [链接](configs/ppyolo/ppyolo_r50vd_dcn_2x_coco.yml) | [下载地址](https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_2x_coco.pdparams) |
| PP-YOLOE-l |     50.9      |    149.2    |    [链接](configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml) | [下载地址](https://paddledet.bj.bcebos.com/models/ppyolov2_r50vd_dcn_365e_coco.pdparams) |
| PP-YOLOE-x |     51.9      |    95.2    |    [链接](configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml) | [下载地址](https://paddledet.bj.bcebos.com/models/ppyolov2_r50vd_dcn_365e_coco.pdparams) |


#### PP-PicoDet系列 推荐场景：ARM CPU移动端芯片和x86 CPU设备

| 模型名称   |   COCO精度（mAP）| 骁龙865 四线程速度(ms) |  配置文件  |   模型下载  |
| :-------- | :---------------: | :------------: |:-------: |:-------: |
| PicoDet-XS |     23.5      |    7.81    |    [链接](configs/picodet/picodet_xs_320_coco_lcnet.yml) | [下载地址](https://paddledet.bj.bcebos.com/models/picodet_xs_320_coco_lcnet.pdparams) |
| PicoDet-S |     29.1      |    9.56    |    [链接](configs/picodet/picodet_s_320_coco_lcnet.yml) | [下载地址](https://paddledet.bj.bcebos.com/models/picodet_s_320_coco_lcnet.pdparams) |
| PicoDet-M |     34.4      |    17.68    |    [链接](configs/picodet/picodet_m_320_coco_lcnet.yml) | [下载地址](https://paddledet.bj.bcebos.com/models/picodet_m_320_coco_lcnet.pdparams) |
| PicoDet-L |     36.1      |    25.21    |    [链接](configs/picodet/picodet_l_320_coco_lcnet.yml) | [下载地址](https://paddledet.bj.bcebos.com/models/picodet_l_320_coco_lcnet.pdparams) |


#### 前沿检测算法

| 模型名称   |   COCO精度（mAP）| V100 TensorRT FP16速度(FPS) | 配置文件  |   模型下载  |
| :-------- | :---------------: |:-------: |:-------: |:-------: |
| YOLOX-l |   50.1    | 107.5 |  [链接](configs/yolox/yolox_l_300e_coco.yml) | [下载地址](https://paddledet.bj.bcebos.com/models/yolox_l_300e_coco.pdparams) |
| YOLOv5-l |   48.6  | 136.0  |  [链接](https://github.com/nemonameless/PaddleDetection_YOLOv5/blob/main/configs/yolov5/yolov5_l_300e_coco.yml) | [下载地址](https://paddledet.bj.bcebos.com/models/yolov5_l_300e_coco.pdparams) |


#### 其他通用检测模型 [文档链接](docs/MODEL_ZOO_cn.md)

### 2. 实例分割

| 模型名称   |  模型简介 |  推荐场景  |  COCO精度(mAP) |  配置文件  |   模型下载  |
| :-------- |  :-------- | :-------- |:---------------: | :------------: |:-------: |
| Mask RCNN |   两阶段实例分割算法 | 服务器端 |  box AP: 41.4 <br/> mask AP: 37.5    |    [链接](configs/mask_rcnn/mask_rcnn_r50_vd_fpn_2x_coco.yml) | [下载地址](https://paddledet.bj.bcebos.com/models/mask_rcnn_r50_vd_fpn_2x_coco.pdparams)
| Cascade Mask RCNN |     两阶段实例分割算法     |  服务器端 |  box AP: 45.7 <br/> mask AP: 39.7    |    [链接](configs/mask_rcnn/cascade_mask_rcnn_r50_vd_fpn_ssld_2x_coco.yml) | [下载地址](https://paddledet.bj.bcebos.com/models/cascade_mask_rcnn_r50_vd_fpn_ssld_2x_coco.pdparams) |
| SOLOv2 |     轻量级单阶段实例分割算法     |  服务器端 |  mask AP: 38.0    |    [链接](configs/solov2/solov2_r50_fpn_3x_coco.yml) | [下载地址](https://paddledet.bj.bcebos.com/models/solov2_r50_fpn_3x_coco.pdparams) |


### 3. 关键点检测

| 模型名称   |  模型简介 |  推荐场景  |  COCO精度（AP）|  速度 | 配置文件  |   模型下载  |
| :-------- |  :-------- | :-------- |:---------------: |:-------: | :------------: |:-------: |
| DARK_HRNet-w32 + DarkPose |   top-down关键点检测算法<br/>输入尺寸384x288 | 服务器端 |  78.3    |  T4 TensorRT FP16 2.96ms      |   [链接](configs/keypoint/hrnet/dark_hrnet_w32_384x288.yml) | [下载地址](https://paddledet.bj.bcebos.com/models/keypoint/dark_hrnet_w32_384x288.pdparams)
| HRNet-w32 + DarkPose |     top-down关键点检测算法<br/>输入尺寸256x192     |  服务器端 |  78.0   |  T4 TensorRT FP16 1.75ms |   [链接](configs/keypoint/hrnet/dark_hrnet_w32_256x192.yml) | [下载地址](https://paddledet.bj.bcebos.com/models/keypoint/dark_hrnet_w32_256x192.pdparams) |
| PP-TinyPose |     轻量级关键点算法<br/>输入尺寸256x192    |  移动端 |  68.8   | 骁龙865 四线程 6.30ms   |  [链接](configs/keypoint/tiny_pose/tinypose_256x192.yml) | [下载地址](https://bj.bcebos.com/v1/paddledet/models/keypoint/tinypose_256x192.pdparams) |
| PP-TinyPose |     轻量级关键点算法<br/>输入尺寸128x96     |  移动端 |  58.1    |  骁龙865 四线程 2.37ms  | [链接](configs/keypoint/tiny_pose/tinypose_128x96.yml) | [下载地址](https://bj.bcebos.com/v1/paddledet/models/keypoint/tinypose_128x96.pdparams) |

#### 其他关键点检测模型 [文档链接](configs/keypoint)


### 4. 多目标跟踪PP-Tracking

| 模型名称   |  模型简介 |  推荐场景  |  精度|  配置文件  |   模型下载  |
| :-------- |  :-------- | :-------- |:---------------: | :------------: |:-------: |
| DeepSORT |   SDE多目标跟踪算法，检测、ReID模型相互独立  | 服务器端 |  MOT-17 half val:  66.9  |    [链接](configs/mot/deepsort/deepsort_jde_yolov3_pcb_pyramid.yml) | [下载地址](https://paddledet.bj.bcebos.com/models/mot/deepsort/deepsort_pcb_pyramid_r101.pdparams)
| ByteTrack |   SDE多目标跟踪算法，仅包含检测模型   |  服务器端 |  MOT-17 half val:  77.3    |    [链接](configs/mot/bytetrack/detector/yolox_x_24e_800x1440_mix_det.yml) | [下载地址](https://paddledet.bj.bcebos.com/models/mot/deepsort/yolox_x_24e_800x1440_mix_det.pdparams) |
| JDE |     JDE多目标跟踪算法，多任务联合学习方法     |  服务器端 |  MOT-16 test: 64.6    |    [链接](configs/mot/jde/jde_darknet53_30e_1088x608.yml) | [下载地址](https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_1088x608.pdparams) |
| FairMOT |     JDE多目标跟踪算法，多任务联合学习方法     |  服务器端 |  MOT-16 test: 75.0    |    [链接](configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml) | [下载地址](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608.pdparams) |

#### 其他多目标跟踪模型 [文档链接](configs/mot)

### 5. 产业级实时行人分析工具PP-Human

| 功能名称   |  适用场景 |  涉及模型 |  精度 |   T4 TensorRT FP16速度(ms) |  模型下载  |
| :-------- |  :-------- | :-------- |:---------------: | :------------: |:-------: |
| 行人检测 |  图片输入  | PP-YOLOE |  mAP 56.3 |   28.0  | [下载地址](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)|
| 行人跟踪 |  视频输入  | PP-YOLOE |  MOTA 72.0 |   33.1  | [下载地址](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip) |
| 属性识别 |  图片/视频输入  | StrongBaseline |  mA 94.86 |   单人2ms  | [下载地址](https://bj.bcebos.com/v1/paddledet/models/pipeline/strongbaseline_r50_30e_pa100k.zip) |
| 行为识别 |  视频输入 摔倒检测  | 关键点检测HRNet<br/> 行为识别ST-GCN |  关键点检测 AP 87.1<br/> 行为识别关键点检测HRNet<br/> 行为识别ST-GCN AP 96.43 | 关键点检测 单人2.9ms<br/> 行为识别 单人2.7ms | 关键点检测 [下载地址](https://bj.bcebos.com/v1/paddledet/models/pipeline/dark_hrnet_w32_256x192.zip) <br/> 行为识别 [下载地址](https://bj.bcebos.com/v1/paddledet/models/pipeline/STGCN.zip) |
| 跨镜跟踪 |  多镜头视频输入  | Centroid |  mAP 98.8 |   单人1.5ms | [下载地址](https://bj.bcebos.com/v1/paddledet/models/pipeline/reid_model.zip) |



## <img src="https://user-images.githubusercontent.com/48054808/157828296-d5eb0ccb-23ea-40f5-9957-29853d7d13a9.png" width="20"/> 文档教程

### 入门教程

- [安装说明](docs/tutorials/INSTALL_cn.md)
- [数据准备](docs/tutorials/data/README.md)
- [快速开始](docs/tutorials/QUICK_STARTED_cn.md)
- [30分钟上手PaddleDetecion](docs/tutorials/GETTING_STARTED_cn.md)
- [FAQ/常见问题汇总](docs/tutorials/FAQ)

### 进阶教程

- 参数配置

  - [RCNN参数说明](docs/tutorials/config_annotation/faster_rcnn_r50_fpn_1x_coco_annotation.md)
  - [PP-YOLO参数说明](docs/tutorials/config_annotation/ppyolo_r50vd_dcn_1x_coco_annotation.md)

- 模型压缩(基于[PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim))

  - [剪裁/量化/蒸馏教程](configs/slim)

- [推理部署](deploy/README.md)

  - [模型导出教程](deploy/EXPORT_MODEL.md)
  - [Paddle Inference部署](deploy/README.md)
    - [Python端推理部署](deploy/python)
    - [C++端推理部署](deploy/cpp)
  - [Paddle-Lite部署](deploy/lite)
  - [Paddle Serving部署](deploy/serving)
  - [ONNX模型导出](deploy/EXPORT_ONNX_MODEL.md)
  - [推理benchmark](deploy/BENCHMARK_INFER.md)

- 进阶开发

  - [数据处理模块](docs/advanced_tutorials/READER.md)
  - [新增检测模型](docs/advanced_tutorials/MODEL_TECHNICAL.md)
  - 二次开发教程
    - [目标检测](docs/advanced_tutorials/customization/detection.md)
    - [关键点检测](docs/advanced_tutorials/customization/keypoint_detection.md)
    - [多目标跟踪](docs/advanced_tutorials/customization/mot.md)
    - [行为识别](docs/advanced_tutorials/customization/action.md)
    - [属性识别](docs/advanced_tutorials/customization/attribute.md)


### 课程专栏

- **2022.4.19 [产业级目标检测技术与应用](https://aistudio.baidu.com/aistudio/education/group/info/23670)三日课：** 超强目标检测算法矩阵、实时行人分析系统PP-Human、目标检测产业应用全流程拆解与实践

- **2022.3.26 [智慧城市行业](https://aistudio.baidu.com/aistudio/education/group/info/25620)七日课：** 城市规划、城市治理、智慧政务、交通管理、社区治理

### [产业实践范例教程](./industrial_tutorial/README_cn.md)

- [基于PP-PicoDet增强版的路面垃圾检测](https://aistudio.baidu.com/aistudio/projectdetail/3846170?channelType=0&channel=0)

- [基于PP-PicoDet的通信塔识别及Android端部署](https://aistudio.baidu.com/aistudio/projectdetail/3561097)

- [基于Faster-RCNN的瓷砖表面瑕疵检测](https://aistudio.baidu.com/aistudio/projectdetail/2571419)

- [基于PaddleDetection的PCB瑕疵检测](https://aistudio.baidu.com/aistudio/projectdetail/2367089)

- [基于FairMOT实现人流量统计](https://aistudio.baidu.com/aistudio/projectdetail/2421822)

- [基于YOLOv3实现跌倒检测 ](https://aistudio.baidu.com/aistudio/projectdetail/2500639)

- [基于PP-PicoDetv2 的路面垃圾检测](https://aistudio.baidu.com/aistudio/projectdetail/3846170?channelType=0&channel=0)

- [基于人体关键点检测的合规检测](https://aistudio.baidu.com/aistudio/projectdetail/4061642?contributionType=1)


## <img title="" src="https://user-images.githubusercontent.com/48054808/157836473-1cf451fa-f01f-4148-ba68-b6d06d5da2f9.png" alt="" width="20"> 应用案例

- [人像圣诞特效自动生成工具](static/application/christmas)
- [安卓健身APP](https://github.com/zhiboniu/pose_demo_android)

## <img src="https://user-images.githubusercontent.com/48054808/160552806-496dc3ba-beb6-4623-8e26-44416b5848bf.png" width="25"/> 第三方教程推荐

- [PaddleDetection在Windows下的部署(一)](https://zhuanlan.zhihu.com/p/268657833)
- [PaddleDetection在Windows下的部署(二)](https://zhuanlan.zhihu.com/p/280206376)
- [Jetson Nano上部署PaddleDetection经验分享](https://zhuanlan.zhihu.com/p/319371293)
- [安全帽检测YOLOv3模型在树莓派上的部署](https://github.com/PaddleCV-FAQ/PaddleDetection-FAQ/blob/main/Lite%E9%83%A8%E7%BD%B2/yolov3_for_raspi.md)
- [使用SSD-MobileNetv1完成一个项目--准备数据集到完成树莓派部署](https://github.com/PaddleCV-FAQ/PaddleDetection-FAQ/blob/main/Lite%E9%83%A8%E7%BD%B2/ssd_mobilenet_v1_for_raspi.md)

## <img src="https://user-images.githubusercontent.com/48054808/157835981-ef6057b4-6347-4768-8fcc-cd07fcc3d8b0.png" width="20"/> 版本更新

版本更新内容请参考[版本更新文档](docs/CHANGELOG.md)

## <img title="" src="https://user-images.githubusercontent.com/48054808/157835345-f5d24128-abaf-4813-b793-d2e5bdc70e5a.png" alt="" width="20"> 许可证书

本项目的发布受[Apache 2.0 license](LICENSE)许可认证。

## <img src="https://user-images.githubusercontent.com/48054808/157835796-08d4ffbc-87d9-4622-89d8-cf11a44260fc.png" width="20"/> 贡献代码

我们非常欢迎你可以为PaddleDetection提供代码，也十分感谢你的反馈。

- 感谢[Mandroide](https://github.com/Mandroide)清理代码并且统一部分函数接口。
- 感谢[FL77N](https://github.com/FL77N/)贡献`Sparse-RCNN`模型。
- 感谢[Chen-Song](https://github.com/Chen-Song)贡献`Swin Faster-RCNN`模型。
- 感谢[yangyudong](https://github.com/yangyudong2020), [hchhtc123](https://github.com/hchhtc123) 开发PP-Tracking GUI界面
- 感谢[Shigure19](https://github.com/Shigure19) 开发PP-TinyPose健身APP
- 感谢[manangoel99](https://github.com/manangoel99)贡献Wandb可视化方式

## <img src="https://user-images.githubusercontent.com/48054808/157835276-9aab9d1c-1c46-446b-bdd4-5ab75c5cfa48.png" width="20"/> 引用

```
@misc{ppdet2019,
title={PaddleDetection, Object detection and instance segmentation toolkit based on PaddlePaddle.},
author={PaddlePaddle Authors},
howpublished = {\url{https://github.com/PaddlePaddle/PaddleDetection}},
year={2019}
}
```
