# PaddleDetection

**注意：** PaddleDetection动态图版本为试用版本，模型广度、模型性能、文档、易用性和兼容性持续优化中，性能数据待发布。


# 简介

PaddleDetection飞桨目标检测开发套件，旨在帮助开发者更快更好地完成检测模型的组建、训练、优化及部署等全开发流程。

PaddleDetection模块化地实现了多种主流目标检测算法，提供了丰富的数据增强策略、网络模块组件（如骨干网络）、损失函数等，并集成了模型压缩和跨平台高性能部署能力。

经过长时间产业实践打磨，PaddleDetection已拥有顺畅、卓越的使用体验，被工业质检、遥感图像检测、无人巡检、新零售、互联网、科研等十多个行业的开发者广泛应用。

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
            <li>PSS-Det RCNN</li>
          </ul>
        </ul>
        <ul><li><b>One-Stage Detection</b></li>
          <ul>
            <li>YOLOv3</li>
            <li>PP-YOLO</li>
            <li>SSD</li>
          </ul>
        </ul>
        <ul><li><b>Anchor Free</b></li>
          <ul>
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
             <li>BlazeFace</li>
            </ul>
        </ul>
      </td>
      <td>
        <ul>
          <li>ResNet(&vd)</li>
          <li>HRNet</li>
          <li>DarkNet</li>
          <li>VGG</li>
          <li>MobileNetv1/v3</li>  
        </ul>
      </td>
      <td>
        <ul><li><b>Common</b></li>
          <ul>
            <li>Sync-BN</li>
            <li>DCNv2</li>
          </ul>  
        </ul>
        <ul><li><b>Loss</b></li>
          <ul>
            <li>Smooth-L1 Loss</li>
            <li>IoU Loss</li>  
            <li>IoU Aware Loss</li>
          </ul>  
        </ul>  
        <ul><li><b>Post-processing</b></li>
          <ul>
            <li>SoftNMS</li>
            <li>MatrixNMS</li>  
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


### 扩展特性

- [√] **Synchronized Batch Norm**
- [√] **Modulated Deformable Convolution**
- [x] **Group Norm**
- [x] **Deformable PSRoI Pooling**

## 文档教程

### 入门教程

- [安装说明](docs/tutorials/INSTALL_cn.md)
- [快速开始](docs/tutorials/QUICK_STARTED_cn.md)
- [如何准备数据](docs/tutorials/PrepareDataSet.md)
- [训练/评估/预测流程](docs/tutorials/GETTING_STARTED_cn.md)

### 进阶教程

- [模型压缩](configs/slim)
- [推理部署](deploy)
    - [模型导出教程](deploy/EXPORT_MODEL.md)
    - [Python端推理部署](deploy/python)
    - [C++端推理部署](deploy/cpp)
    - [服务端部署](deploy/serving)
- [进阶开发]
    - [数据处理模块](docs/advanced_tutorials/READER.md)
    - [新增检测模型](docs/advanced_tutorials/MODEL_TECHNICAL.md)


## 模型库

- 通用目标检测:
    - [模型库](docs/MODEL_ZOO_cn.md)
- 垂类领域:
    - [行人检测](configs/pedestrian/README.md)
    - [车辆检测](configs/vehicle/README.md)


## 许可证书

本项目的发布受[Apache 2.0 license](LICENSE)许可认证。


## 贡献代码

我们非常欢迎你可以为PaddleDetection提供代码，也十分感谢你的反馈。
