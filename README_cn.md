简体中文 | [English](README_en.md)

# PaddleDetection

### PaddleDetection 2.0全面升级！目前默认使用动态图版本，静态图版本位于[static](./static)中
### 超高性价比PPYOLO v2和1.3M超轻量PPYOLO tiny全新出炉！[欢迎使用](configs/ppyolo/README_cn.md)
### Anchor Free SOTA模型PAFNet发布！[欢迎使用](configs/ttfnet/README.md)
# 近期活动

百度飞桨产业级目标检测技术详解系列直播课，看超越YOLOv5的PP-YOLOv2到底多强大

欢迎大家加入PPYOLOv2&Tiny技术交流群


<div align="left">
  <img src="https://z3.ax1x.com/2021/05/11/gUDw0e.png" width='150'/>
</div>


### 课程安排     
 [直播链接](http://live.bilibili.com/21689802)
* 5月13日19:00-20:00
  -  主题: 产业级目标检测算法全解读
* 5月14日19:00-20:00
   - 主题: 1.3M超轻量目标检测算法解读及应用
* 5月21日20:00-21:00
   - 主题: 复杂背景下小目标检测模型开发实战

### 学习链接

 [0【PaddleDetection2.0专项】新版本快速体验](https://aistudio.baidu.com/aistudio/projectdetail/1885319)

 [1【PaddleDetection2.0专项】如何自定义数据集](https://aistudio.baidu.com/aistudio/projectdetail/1917140)

 [2【PaddleDetection2.0专项】快速上手PP-YOLOv2](https://aistudio.baidu.com/aistudio/projectdetail/1922155)

 [3【PaddleDetection2.0专项】快速上手PP-YOLO tiny](https://aistudio.baidu.com/aistudio/projectdetail/1918450)

 [4【PaddleDetection2.0专项】快速上手S2ANet](https://aistudio.baidu.com/aistudio/projectdetail/1923957)

 [5【PaddleDetection2.0专项】快速实现行人检测](https://aistudio.baidu.com/aistudio/projectdetail/1918451)

 [6【PaddleDetection2.0专项】快速实现人脸检测](https://aistudio.baidu.com/aistudio/projectdetail/1918453)


# 简介

PaddleDetection飞桨目标检测开发套件，旨在帮助开发者更快更好地完成检测模型的组建、训练、优化及部署等全开发流程。

PaddleDetection模块化地实现了多种主流目标检测算法，提供了丰富的数据增强策略、网络模块组件（如骨干网络）、损失函数等，并集成了模型压缩和跨平台高性能部署能力。

经过长时间产业实践打磨，PaddleDetection已拥有顺畅、卓越的使用体验，被工业质检、遥感图像检测、无人巡检、新零售、互联网、科研等十多个行业的开发者广泛应用。

<div align="center">
  <img src="static/docs/images/football.gif" width='800'/>
</div>

### 产品动态
- 2021.04.14: 发布release/2.0版本，PaddleDetection全面支持动态图，覆盖静态图模型算法，全面升级模型效果，同时发布[PP-YOLO v2, PPYOLO tiny](configs/ppyolo/README_cn.md)模型，增强版anchor free模型[PAFNet](configs/ttfnet/README.md)，新增旋转框检测[S2ANet](configs/dota/README.md)模型，详情参考[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.0)
- 2021.02.07: 发布release/2.0-rc版本，PaddleDetection动态图试用版本，详情参考[PaddleDetection动态图](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.0-rc)。

### 特性

- **模型丰富**: 包含**目标检测**、**实例分割**、**人脸检测**等**100+个预训练模型**，涵盖多种**全球竞赛冠军**方案
- **使用简洁**：模块化设计，解耦各个网络组件，开发者轻松搭建、试用各种检测模型及优化策略，快速得到高性能、定制化的算法。
- **端到端打通**: 从数据增强、组网、训练、压缩、部署端到端打通，并完备支持**云端**/**边缘端**多架构、多设备部署。
- **高性能**: 基于飞桨的高性能内核，模型训练速度及显存占用优势明显。支持FP16训练, 支持多机训练。


### 套件结构概览

<table>
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
        <ul><li><b>Two-Stage Detection</b></li>
          <ul>
            <li>Faster RCNN</li>
            <li>FPN</li>
            <li>Cascade-RCNN</li>
            <li>Libra RCNN</li>
            <li>Hybrid Task RCNN</li>
            <li>PSS-Det</li>
          </ul>
        </ul>
        <ul><li><b>One-Stage Detection</b></li>
          <ul>
            <li>RetinaNet</li>
            <li>YOLOv3</li>
            <li>YOLOv4</li>  
            <li>PP-YOLO</li>
            <li>SSD</li>
          </ul>
        </ul>
        <ul><li><b>Anchor Free</b></li>
          <ul>
            <li>CornerNet-Squeeze</li>
            <li>FCOS</li>  
            <li>TTFNet</li>
          </ul>
        </ul>
        <ul>
          <li><b>Instance Segmentation</b></li>
            <ul>
             <li>Mask RCNN</li>
             <li>SOLOv2</li>
            </ul>
        </ul>
        <ul>
          <li><b>Face-Detction</b></li>
            <ul>
             <li>FaceBoxes</li>
             <li>BlazeFace</li>
             <li>BlazeFace-NAS</li>
            </ul>
        </ul>
      </td>
      <td>
        <ul>
          <li>ResNet(&vd)</li>
          <li>ResNeXt(&vd)</li>
          <li>SENet</li>
          <li>Res2Net</li>
          <li>HRNet</li>
          <li>Hourglass</li>
          <li>CBNet</li>
          <li>GCNet</li>
          <li>DarkNet</li>
          <li>CSPDarkNet</li>
          <li>VGG</li>
          <li>MobileNetv1/v3</li>  
          <li>GhostNet</li>
          <li>Efficientnet</li>  
        </ul>
      </td>
      <td>
        <ul><li><b>Common</b></li>
          <ul>
            <li>Sync-BN</li>
            <li>Group Norm</li>
            <li>DCNv2</li>
            <li>Non-local</li>
          </ul>  
        </ul>
        <ul><li><b>FPN</b></li>
          <ul>
            <li>BiFPN</li>
            <li>BFP</li>  
            <li>HRFPN</li>
            <li>ACFPN</li>
          </ul>  
        </ul>  
        <ul><li><b>Loss</b></li>
          <ul>
            <li>Smooth-L1</li>
            <li>GIoU/DIoU/CIoU</li>  
            <li>IoUAware</li>
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
          <li>Flipping</li>  
          <li>Expand</li>
          <li>Crop</li>
          <li>Color Distort</li>  
          <li>Random Erasing</li>  
          <li>Mixup </li>
          <li>Cutmix </li>
          <li>Grid Mask</li>
          <li>Auto Augment</li>  
        </ul>  
      </td>  
    </tr>


</td>
    </tr>
  </tbody>
</table>

#### 模型性能概览

各模型结构和骨干网络的代表模型在COCO数据集上精度mAP和单卡Tesla V100上预测速度(FPS)对比图。

<div align="center">
  <img src="docs/images/fps_map.png" />
</div>

**说明：**

- `CBResNet`为`Cascade-Faster-RCNN-CBResNet200vd-FPN`模型，COCO数据集mAP高达53.3%
- `Cascade-Faster-RCNN`为`Cascade-Faster-RCNN-ResNet50vd-DCN`，PaddleDetection将其优化到COCO数据mAP为47.8%时推理速度为20FPS
- `PP-YOLO`在COCO数据集精度45.9%，Tesla V100预测速度72.9FPS，精度速度均优于[YOLOv4](https://arxiv.org/abs/2004.10934)
- `PP-YOLO v2`是对`PP-YOLO`模型的进一步优化，在COCO数据集精度49.5%，Tesla V100预测速度68.9FPS
- 图中模型均可在[模型库](#模型库)中获取

## 文档教程

### 入门教程

- [安装说明](docs/tutorials/INSTALL_cn.md)
- [快速开始](docs/tutorials/QUICK_STARTED_cn.md)
- [如何准备数据](docs/tutorials/PrepareDataSet.md)
- [训练/评估/预测流程](docs/tutorials/GETTING_STARTED_cn.md)

### 进阶教程

- 参数配置
    - [RCNN参数说明](docs/tutorials/config_annotation/faster_rcnn_r50_fpn_1x_coco_annotation.md)
    - [PP-YOLO参数说明](docs/tutorials/config_annotation/ppyolo_r50vd_dcn_1x_coco_annotation.md)
- 模型压缩(基于[PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim))
    - [剪裁/量化/蒸馏教程](configs/slim)
- [推理部署](deploy/README.md)
    - [模型导出教程](deploy/EXPORT_MODEL.md)
    - [Python端推理部署](deploy/python)
    - [C++端推理部署](deploy/cpp)
    - [服务端部署](deploy/serving)
    - [推理benchmark](deploy/BENCHMARK_INFER.md)
- 进阶开发
    - [数据处理模块](docs/advanced_tutorials/READER.md)
    - [新增检测模型](docs/advanced_tutorials/MODEL_TECHNICAL.md)


## 模型库

- 通用目标检测:
    - [模型库](docs/MODEL_ZOO_cn.md)
    - [PP-YOLO模型](configs/ppyolo/README_cn.md)
    - [增强版Anchor Free模型TTFNet](configs/ttfnet/README.md)
    - [移动端模型](static/configs/mobile/README.md)
    - [676类目标检测](static/docs/featured_model/LARGE_SCALE_DET_MODEL.md)
    - [两阶段实用模型PSS-Det](configs/rcnn_enhance/README.md)
    - [半监督知识蒸馏预训练检测模型](docs/feature_models/SSLD_PRETRAINED_MODEL.md)
- 通用实例分割
    - [SOLOv2](configs/solov2/README.md)
- 旋转框检测
    - [S2ANet](configs/dota/README.md)
- 垂类领域
    - [行人检测](configs/pedestrian/README.md)
    - [车辆检测](configs/vehicle/README.md)
    - [人脸检测](configs/face_detection/README.md)
- 比赛冠军方案
    - [Objects365 2019 Challenge夺冠模型](static/docs/featured_model/champion_model/CACascadeRCNN.md)
    - [Open Images 2019-Object Detction比赛最佳单模型](static/docs/featured_model/champion_model/OIDV5_BASELINE_MODEL.md)

## 应用案例

- [人像圣诞特效自动生成工具](static/application/christmas)

## 第三方教程推荐

- [PaddleDetection在Windows下的部署(一)](https://zhuanlan.zhihu.com/p/268657833)
- [PaddleDetection在Windows下的部署(二)](https://zhuanlan.zhihu.com/p/280206376)
- [Jetson Nano上部署PaddleDetection经验分享](https://zhuanlan.zhihu.com/p/319371293)
- [安全帽检测YOLOv3模型在树莓派上的部署](https://github.com/PaddleCV-FAQ/PaddleDetection-FAQ/blob/main/Lite%E9%83%A8%E7%BD%B2/yolov3_for_raspi.md)
- [使用SSD-MobileNetv1完成一个项目--准备数据集到完成树莓派部署](https://github.com/PaddleCV-FAQ/PaddleDetection-FAQ/blob/main/Lite%E9%83%A8%E7%BD%B2/ssd_mobilenet_v1_for_raspi.md)

## 版本更新

v2.0版本已经在`04/2021`发布，全面支持动态图版本，新增支持BlazeFace, PSSDet等系列模型和大量骨干网络，发布PP-YOLO v2, PP-YOLO tiny和旋转框检测S2ANet模型。支持模型蒸馏、VisualDL，新增动态图预测部署benchmark，详细内容请参考[版本更新文档](docs/CHANGELOG.md)。


## 许可证书

本项目的发布受[Apache 2.0 license](LICENSE)许可认证。


## 贡献代码

我们非常欢迎你可以为PaddleDetection提供代码，也十分感谢你的反馈。


## 引用

```
@misc{ppdet2019,
title={PaddleDetection, Object detection and instance segmentation toolkit based on PaddlePaddle.},
author={PaddlePaddle Authors},
howpublished = {\url{https://github.com/PaddlePaddle/PaddleDetection}},
year={2019}
}
```
