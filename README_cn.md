## 简介

PaddleDetection是一个基于PaddlePaddle的目标检测端到端开发套件。

<img src="https://paddledet.bj.bcebos.com/docs/images/ppdet.gif"/>

### 主要特性

- **模块化设计**
PaddleDetection将检测模型解耦成不同的模块组件，通过自定义模块组件组合，用户可以便捷高效地完成检测模型的搭建。`传送门`：[模块组件](#模块组件)。

- **丰富的模型库**
PaddleDetection支持大量的最新主流的算法基准以及预训练模型，涵盖目标检测、实例分割、人脸检测、关键点检测、多目标跟踪等方向。传送门：[]()。

- **产业级特色模型&分析工具**
PaddleDetection打造产业级特色模型以及分析工具：PP-YOLOE+、PP-PicoDet、PP-TinyPose、PP-HumanV2、PP-Vehicle等，针对通用、高频垂类应用场景提供深度优化解决方案以及高度集成的分析工具，降低开发者的试错、选择成本，针对业务场景快速应用落地。`传送门`：[产业实践范例](#产业实践范例)。

- **产业级部署实践**
PaddleDetection整理工业、农业、林业、交通、医疗、金融、能源电力等AI应用范例，打通数据标注-模型训练-模型调优-预测部署全流程，持续降低目标检测技术产业落地门槛。`传送门`：[产业实践范例](#产业实践范例)、[企业应用案例](#企业应用案例)。

## 最新进展

### 稳定版本

位于[`release/2.5`](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5)分支，最新的[**v2.5**](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5)版本已经在 2022.09.13 发布，版本发新内容请参考[v2.5.0更新日志](https://github.com/PaddlePaddle/PaddleDetection/releases/tag/v2.5.0)

### 预览版本

位于[`develop`](https://github.com/PaddlePaddle/PaddleDetection/tree/develop)分支，体验最新功能请切换到[该分支](https://github.com/PaddlePaddle/PaddleDetection/tree/develop)：
- **模型库**
  - 新增[半监督检测模型](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/semi_det);
- **产业级特色模型**
  - 发布**旋转框检测模型**[PP-YOLOE-R](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/rotate/ppyoloe_r)：Anchor-free旋转框检测SOTA模型，精度速度双高、云边一体，s/m/l/x四个模型适配不用算力硬件、部署友好，避免使用特殊算子，能够轻松使用TensorRT加速；
  - 发布**小目标检测模型**[PP-YOLOE-SOD](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/smalldet)：基于切图的端到端检测方案、基于原图的检测模型，精度达VisDrone开源最优；

### 近期技术分享

- **【AI快车道两日课】手把手教你将PP-YOLOE+用于旋转框、小目标检测，达成SOTA性能**
  - **详情点击**：
  [Yes, PP-YOLOE！80.73mAP、38.5mAP，旋转框、小目标检测能力双SOTA！](https://mp.weixin.qq.com/s/6ji89VKqoXDY6SSGkxS8NQ)
  - **扫码入群即可获取直播课程&PPT链接与技术大礼包！**
  <div align="center">
  <img src="https://user-images.githubusercontent.com/22989727/202123813-1097e3f6-c784-4991-9b94-8cbcd972de82.png"  width = "200" />  
  </div>

## 安装

参考[安装说明](docs/tutorials/INSTALL_cn.md)进行安装。

## 教程
### 深度学习入门教程

- [零基础入门深度学习](https://www.paddlepaddle.org.cn/tutorials/projectdetail/4676538)
- [零基础入门目标检测](https://aistudio.baidu.com/aistudio/education/group/info/1617)
### 快速开始

- [快速体验](docs/tutorials/QUICK_STARTED_cn.md)
- [示例：30分钟快速开发交通标志检测模型](docs/tutorials/GETTING_STARTED_cn.md)

### 数据准备
- [数据准备](docs/tutorials/data/README.md)
- [数据处理模块](docs/advanced_tutorials/READER.md)
### 配置文件说明
- [RCNN参数说明](docs/tutorials/config_annotation/faster_rcnn_r50_fpn_1x_coco_annotation.md)
- [PP-YOLO参数说明](docs/tutorials/config_annotation/ppyolo_r50vd_dcn_1x_coco_annotation.md)

### 模型开发

- [新增检测模型](docs/advanced_tutorials/MODEL_TECHNICAL.md)
- 二次开发
  - [目标检测](docs/advanced_tutorials/customization/detection.md)
  - [关键点检测](docs/advanced_tutorials/customization/keypoint_detection.md)
  - [多目标跟踪](docs/advanced_tutorials/customization/pphuman_mot.md)
  - [行为识别](docs/advanced_tutorials/customization/action_recognotion/)
  - [属性识别](docs/advanced_tutorials/customization/pphuman_attribute.md)

### 部署推理
- [模型导出教程](deploy/EXPORT_MODEL.md)
- 模型压缩(基于[PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim))
- [剪裁/量化/蒸馏教程](configs/slim)
- [推理部署](deploy/README.md)
- [Paddle Inference部署](deploy/README.md)
  - [Python端推理部署](deploy/python)
  - [C++端推理部署](deploy/cpp)
- [Paddle-Lite部署](deploy/lite)
- [Paddle Serving部署](deploy/serving)
- [ONNX模型导出](deploy/EXPORT_ONNX_MODEL.md)
- [推理benchmark](deploy/BENCHMARK_INFER.md)

## 模块组件

<table align="center">
  <tbody>
    <tr align="center" valign="center">
      <td>
        <b>Backbones</b>
      </td>
      <td>
        <b>Necks</b>
      </td>
      <td>
        <b>Loss</b>
      </td>
      <td>
        <b>Common</b>
      </td>
      <td>
      <b>Data Augmentation</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
      <ul>
        <li>ResNet(&vd)</li>
          <li>Res2Net(&vd)</li>
          <li>CSPResNet</li>
          <li>SENet</li>
          <li>Res2Net</li>
          <li><a href="configs/hrnet">HRNet</a></li>
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
          <li>ConvNeXt</li>
          <li>Vision Transformer</li>
      </ul>
      </td>
      <td>
      <ul>
        <li>BiFPN</li>
        <li>CSP-PAN</li>
        <li>Custom-PAN</li>
        <li>ES-PAN</li>
        <li>HRFPN</li>
      </ul>
      </td>
      <td>
        <ul>
          <li>Smooth-L1</li>
          <li>GIoU/DIoU/CIoU</li>  
          <li>IoUAware</li>
          <li>Focal Loss</li>
          <li>CT Focal Loss</li>
          <li>VariFocal Loss</li>
        </ul>
      </td>
      <td>
        <ul>
          <li>SoftNMS</li>
            <li>MatrixNMS</li>
            <li>FP16 training</li>
            <li>Multi-machine training </li> 
            <li>Sync-BN</li>
            <li>Group Norm</li>
            <li>DCNv2</li>
            <li>EMA</li>
            <li>DarkPose</li>
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

## 模型库

<table align="center">
  <tbody>
    <tr align="center" valign="center">
      <td>
        <b>Object Detection</b>
      </td>
      <td>
        <b>Instance Segmentation</b>
      </td>
      <td>
        <b>Multi Object Tracking</b>
      </td>
      <td>
        <b>KeyPoint Detection</b>
      </td>
      <td>
      <b>Others</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
            <li>Faster RCNN</li>
            <li>FPN</li>
            <li>Cascade-RCNN</li>
            <li>PSS-Det</li>
            <li>RetinaNet</li>
            <li>YOLOv3</li>  
            <li>YOLOv5</li>  
            <li>YOLOv6</li>  
            <li>YOLOv7</li>  
            <li>PP-YOLOv1/v2</li>
            <li>PP-YOLO-Tiny</li>
            <li>PP-YOLOE</li>
            <li>PP-YOLOE+</li>
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
      </td>
      <td>
        <ul>
          <li>Mask RCNN</li>
            <li>Cascade Mask RCNN</li>
            <li>SOLOv2</li>
        </ul>
      </td>
      <td>
        <ul>
           <li>JDE</li>
            <li>FairMOT</li>
            <li>DeepSORT</li>
            <li>ByteTrack</li>
            <li>OC-SORT</li>
        </ul>
      </td>
      <td>
        <ul>
          <li>HRNet</li>
            <li>HigherHRNet</li>
            <li>Lite-HRNet</li>
            <li>PP-TinyPose</li>
        </ul>
</td>
<td>
      </ul>
          <li><b>Face Detection</b></li>
        <ul>
        <ul>
            <li>BlazeFace</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

## 产业级特色模型&分析工具

### PP-YOLOE 高精度目标检测模型

| 模型名称       | COCO精度（mAP） | V100 TensorRT FP16速度(FPS) | 配置文件                                                  | 模型下载                                                                                 |
|:---------- |:-----------:|:-------------------------:|:-----------------------------------------------------:|:------------------------------------------------------------------------------------:|
| PP-YOLOE+_s | 43.9        | 333.3                     | [链接](configs/ppyoloe/ppyoloe_plus_crn_s_80e_coco.yml)     | [下载地址](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_s_80e_coco.pdparams)      |
| PP-YOLOE+_m | 50.0        | 208.3                     | [链接](configs/ppyoloe/ppyoloe_plus_crn_m_80e_coco.yml)     | [下载地址](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_m_80e_coco.pdparams)     |
| PP-YOLOE+_l | 53.3        | 149.2                     | [链接](configs/ppyoloe/ppyoloe_plus_crn_l_80e_coco.yml) | [下载地址](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_m_80e_coco.pdparams) |
| PP-YOLOE+_x | 54.9        | 95.2                      | [链接](configs/ppyoloe/ppyoloe_plus_crn_x_80e_coco.yml) | [下载地址](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_x_80e_coco.pdparams) |

### PP-PicoDet 超轻量实时目标检测模型

| 模型名称       | COCO精度（mAP） | 骁龙865 四线程速度(ms) | 配置文件                                                | 模型下载                                                                              |
|:---------- |:-----------:|:---------------:|:---------------------------------------------------:|:---------------------------------------------------------------------------------:|
| PicoDet-XS | 23.5        | 7.81            | [链接](configs/picodet/picodet_xs_320_coco_lcnet.yml) | [下载地址](https://paddledet.bj.bcebos.com/models/picodet_xs_320_coco_lcnet.pdparams) |
| PicoDet-S  | 29.1        | 9.56            | [链接](configs/picodet/picodet_s_320_coco_lcnet.yml)  | [下载地址](https://paddledet.bj.bcebos.com/models/picodet_s_320_coco_lcnet.pdparams)  |
| PicoDet-M  | 34.4        | 17.68           | [链接](configs/picodet/picodet_m_320_coco_lcnet.yml)  | [下载地址](https://paddledet.bj.bcebos.com/models/picodet_m_320_coco_lcnet.pdparams)  |
| PicoDet-L  | 36.1        | 25.21           | [链接](configs/picodet/picodet_l_320_coco_lcnet.yml)  | [下载地址](https://paddledet.bj.bcebos.com/models/picodet_l_320_coco_lcnet.pdparams)  |

### PP-Tracking 实时多目标跟踪系统
| 模型名称      | 模型简介                     | 推荐场景                               | 精度                     | 配置文件                                                                  | 模型下载                                                                                              |
|:--------- |:------------------------ |:---------------------------------- |:----------------------:|:---------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------:|
| ByteTrack | SDE多目标跟踪算法 仅包含检测模型       | 云边端                                | MOT-17 test:  78.4 | [链接](configs/mot/bytetrack/bytetrack_yolox.yml) | [下载地址](https://bj.bcebos.com/v1/paddledet/models/mot/yolox_x_24e_800x1440_mix_det.pdparams) |
| FairMOT   | JDE多目标跟踪算法 多任务联合学习方法     | 云边端                                | MOT-16 test: 75.0      | [链接](configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml)              | [下载地址](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608.pdparams)            |
| OC-SORT | SDE多目标跟踪算法 仅包含检测模型       | 云边端                                | MOT-17 half val:  75.5 | [链接](configs/mot/ocsort/ocsort_yolox.yml) | [下载地址](https://bj.bcebos.com/v1/paddledet/models/mot/yolox_x_24e_800x1440_mix_mot_ch.pdparams) |
### PP-TinyPose 人体骨骼关键点识别

| 模型名称                                        | 模型简介                                                             | 推荐场景                               | COCO精度（AP） | 速度                      | 配置文件                                                    | 模型下载                                                                                    |
|:------------------------------------------- |:---------------------------------------------------------------- |:---------------------------------- |:----------:|:-----------------------:|:-------------------------------------------------------:|:---------------------------------------------------------------------------------------:|
| HRNet-w32 + DarkPose                        | <div style="width: 130pt">top-down 关键点检测算法<br/>输入尺寸384x288</div> | <div style="width: 50pt">云边端</div> | 78.3       | T4 TensorRT FP16 2.96ms | [链接](configs/keypoint/hrnet/dark_hrnet_w32_384x288.yml) | [下载地址](https://paddledet.bj.bcebos.com/models/keypoint/dark_hrnet_w32_384x288.pdparams) |
| HRNet-w32 + DarkPose                        | top-down 关键点检测算法<br/>输入尺寸256x192                                 | 云边端                                | 78.0       | T4 TensorRT FP16 1.75ms | [链接](configs/keypoint/hrnet/dark_hrnet_w32_256x192.yml) | [下载地址](https://paddledet.bj.bcebos.com/models/keypoint/dark_hrnet_w32_256x192.pdparams) |
| [PP-TinyPose](./configs/keypoint/tiny_pose) | 轻量级关键点算法<br/>输入尺寸256x192                                         | 移动端                                | 68.8       | 骁龙865 四线程 6.30ms        | [链接](configs/keypoint/tiny_pose/tinypose_256x192.yml)   | [下载地址](https://bj.bcebos.com/v1/paddledet/models/keypoint/tinypose_256x192.pdparams)    |
| [PP-TinyPose](./configs/keypoint/tiny_pose) | 轻量级关键点算法<br/>输入尺寸128x96                                          | 移动端                                | 58.1       | 骁龙865 四线程 2.37ms        | [链接](configs/keypoint/tiny_pose/tinypose_128x96.yml)    | [下载地址](https://bj.bcebos.com/v1/paddledet/models/keypoint/tinypose_128x96.pdparams)     |
### PP-Human 实时行人分析工具
| 任务            | 端到端速度（ms）|  模型方案  |  模型体积 |
| :---------:     | :-------:  |  :------: |:------: |
|  行人检测（高精度）  | 25.1ms  |  [目标检测](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip) | 182M |  
|  行人检测（轻量级）  | 16.2ms  |  [目标检测](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_s_36e_pipeline.zip) | 27M |
|  行人跟踪（高精度）  | 31.8ms  |  [多目标跟踪](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip) | 182M |  
|  行人跟踪（轻量级）  | 21.0ms  |  [多目标跟踪](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_s_36e_pipeline.zip) | 27M |
|  属性识别（高精度）  |   单人8.5ms | [目标检测](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)<br> [属性识别](https://bj.bcebos.com/v1/paddledet/models/pipeline/PPHGNet_small_person_attribute_954_infer.zip) | 目标检测：182M<br>属性识别：86M |
|  属性识别（轻量级）  |   单人7.1ms | [目标检测](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)<br> [属性识别](https://bj.bcebos.com/v1/paddledet/models/pipeline/PPLCNet_x1_0_person_attribute_945_infer.zip) | 目标检测：182M<br>属性识别：86M |
|  摔倒识别  |   单人10ms | [多目标跟踪](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip) <br> [关键点检测](https://bj.bcebos.com/v1/paddledet/models/pipeline/dark_hrnet_w32_256x192.zip) <br> [基于关键点行为识别](https://bj.bcebos.com/v1/paddledet/models/pipeline/STGCN.zip) | 多目标跟踪：182M<br>关键点检测：101M<br>基于关键点行为识别：21.8M |
|  闯入识别  |   31.8ms | [多目标跟踪](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip) | 182M |
|  打架识别  |   19.7ms | [视频分类](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip) | 90M |
|  抽烟识别  |   单人15.1ms | [目标检测](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)<br>[基于人体id的目标检测](https://bj.bcebos.com/v1/paddledet/models/pipeline/ppyoloe_crn_s_80e_smoking_visdrone.zip) | 目标检测：182M<br>基于人体id的目标检测：27M |
|  打电话识别  |   单人ms | [目标检测](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)<br>[基于人体id的图像分类](https://bj.bcebos.com/v1/paddledet/models/pipeline/PPHGNet_tiny_calling_halfbody.zip) | 目标检测：182M<br>基于人体id的图像分类：45M |


PP-Vehicle 实时车辆分析工具
（首个产业级开源）
特色：囊括车牌识别、属性识别、车流量统计、违章检测四大交通场景核心功能，支持自定义二次新增功能开发。


## 产业实践范例

产业实践范例是PaddleDetection针对高频目标检测应用场景，提供的端到端开发示例，帮助开发者打通数据标注-模型训练-模型调优-预测部署全流程。
针对每个范例我们都通过[AI-Studio](https://ai.baidu.com/ai-doc/AISTUDIO/Tk39ty6ho)提供了项目代码以及说明，用户可以同步运行体验。
- [基于PP-YOLOE-R的旋转框检测](https://aistudio.baidu.com/aistudio/projectdetail/5058293)
- [基于PP-YOLOE-SOD的无人机航拍图像检测](https://aistudio.baidu.com/aistudio/projectdetail/5036782)
- [基于PP-Human v2的摔倒检测](https://aistudio.baidu.com/aistudio/projectdetail/4606001)
- [基于PP-TinyPose增强版的智能健身动作识别](https://aistudio.baidu.com/aistudio/projectdetail/4385813)
- [基于PP-Human的打架识别](https://aistudio.baidu.com/aistudio/projectdetail/4086987?contributionType=1)
- [基于PP-PicoDet的通信塔识别及Android端部署](https://aistudio.baidu.com/aistudio/projectdetail/3561097)
- [基于Faster-RCNN的瓷砖表面瑕疵检测](https://aistudio.baidu.com/aistudio/projectdetail/2571419)
- [基于PaddleDetection的PCB瑕疵检测](https://aistudio.baidu.com/aistudio/projectdetail/2367089)
- [基于FairMOT实现人流量统计](https://aistudio.baidu.com/aistudio/projectdetail/2421822)
- [基于YOLOv3实现跌倒检测](https://aistudio.baidu.com/aistudio/projectdetail/2500639)
- [基于PP-PicoDetv2 的路面垃圾检测](https://aistudio.baidu.com/aistudio/projectdetail/3846170?channelType=0&channel=0)
- [基于人体关键点检测的合规检测](https://aistudio.baidu.com/aistudio/projectdetail/4061642?contributionType=1)
- [基于PP-Human的来客分析案例教程](https://aistudio.baidu.com/aistudio/projectdetail/4537344)
持续更新中...

## 企业应用案例
- [中国南方电网——变电站智慧巡检](https://www.paddlepaddle.org.cn/support/news?action=detail&id=2330)
- [国铁电气——轨道在线智能巡检系统](https://www.paddlepaddle.org.cn/support/news?action=detail&id=2280)
- [京东物流——园区车辆行为识别](https://www.paddlepaddle.org.cn/support/news?action=detail&id=2611)
- [中兴克拉—厂区传统仪表统计监测](https://www.paddlepaddle.org.cn/support/news?action=detail&id=2618)
- [宁德时代—动力电池高精度质量检测](https://www.paddlepaddle.org.cn/support/news?action=detail&id=2609)
- [中国科学院空天信息创新研究院——高尔夫球场遥感监测](https://www.paddlepaddle.org.cn/support/news?action=detail&id=2483)
- [御航智能——基于边缘的无人机智能巡检](https://www.paddlepaddle.org.cn/support/news?action=detail&id=2481)
- [普宙无人机——高精度森林巡检](https://www.paddlepaddle.org.cn/support/news?action=detail&id=2121)
- [领邦智能——红外无感测温监控](https://www.paddlepaddle.org.cn/support/news?action=detail&id=2615)
- [北京地铁——口罩检测](https://mp.weixin.qq.com/s/znrqaJmtA7CcjG0yQESWig)
- [音智达——工厂人员违规行为检测](https://www.paddlepaddle.org.cn/support/news?action=detail&id=2288)
- [华夏天信——输煤皮带机器人智能巡检](https://www.paddlepaddle.org.cn/support/news?action=detail&id=2331)
- [优恩物联网——社区住户分类支持广告精准投放](https://www.paddlepaddle.org.cn/support/news?action=detail&id=2485)
- [螳螂慧视——室内3D点云场景物体分割与检测](https://www.paddlepaddle.org.cn/support/news?action=detail&id=2599)
持续更新中...
## FAQ
- [FAQ/常见问题汇总](docs/tutorials/FAQ)
