简体中文 | [English](README_en.md)

<div align="center">
<p align="center">
  <img src="https://user-images.githubusercontent.com/48054808/160532560-34cf7a1f-d950-435e-90d2-4b0a679e5119.png" align="middle" width = "800" />
</p>

<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href="https://github.com/PaddlePaddle/PaddleDetection/releases"><img src="https://img.shields.io/github/v/release/PaddlePaddle/PaddleDetection?color=ffa"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href="https://github.com/PaddlePaddle/PaddleDetection/stargazers"><img src="https://img.shields.io/github/stars/PaddlePaddle/PaddleDetection?color=ccf"></a>
</p>
</div>

## 💌目录
- [💌目录](#目录)
- [🌈简介](#简介)
- [📣最新进展](#最新进展)
- [👫开源社区](#开源社区)
- [✨主要特性](#主要特性)
    - [🧩模块化设计](#模块化设计)
    - [📱丰富的模型库](#丰富的模型库)
    - [🎗️产业特色模型|产业工具](#️产业特色模型产业工具)
    - [💡🏆产业级部署实践](#产业级部署实践)
- [🍱安装](#安装)
- [🔥教程](#教程)
- [🔑FAQ](#faq)
- [🧩模块组件](#模块组件)
- [📱模型库](#模型库)
- [⚖️模型性能对比](#️模型性能对比)
    - [🖥️服务器端模型性能对比](#️服务器端模型性能对比)
    - [⌚️移动端模型性能对比](#️移动端模型性能对比)
- [🎗️产业特色模型|产业工具](#️产业特色模型产业工具-1)
  - [💎PP-YOLOE 高精度目标检测模型](#pp-yoloe-高精度目标检测模型)
  - [💎PP-YOLOE-R 高性能旋转框检测模型](#pp-yoloe-r-高性能旋转框检测模型)
  - [💎PP-YOLOE-SOD 高精度小目标检测模型](#pp-yoloe-sod-高精度小目标检测模型)
  - [💫PP-PicoDet 超轻量实时目标检测模型](#pp-picodet-超轻量实时目标检测模型)
  - [📡PP-Tracking 实时多目标跟踪系统](#pp-tracking-实时多目标跟踪系统)
  - [⛷️PP-TinyPose 人体骨骼关键点识别](#️pp-tinypose-人体骨骼关键点识别)
  - [🏃🏻PP-Human 实时行人分析工具](#pp-human-实时行人分析工具)
  - [🏎️PP-Vehicle 实时车辆分析工具](#️pp-vehicle-实时车辆分析工具)
- [💡产业实践范例](#产业实践范例)
- [🏆企业应用案例](#企业应用案例)
- [📝许可证书](#许可证书)
- [📌引用](#引用)


## 🌈简介

PaddleDetection是一个基于PaddlePaddle的目标检测端到端开发套件，在提供丰富的模型组件和测试基准的同时，注重端到端的产业落地应用，通过打造产业级特色模型|工具、建设产业应用范例等手段，帮助开发者实现数据准备、模型选型、模型训练、模型部署的全流程打通，快速进行落地应用。

主要模型效果示例如下（点击标题可快速跳转）：

|                                                  [**通用目标检测**](#pp-yoloe-高精度目标检测模型)                                                  |                                                [**小目标检测**](#pp-yoloe-sod-高精度小目标检测模型)                                                |                                                  [**旋转框检测**](#pp-yoloe-r-高性能旋转框检测模型)                                                  |                                            [**3D目标物检测**](https://github.com/PaddlePaddle/Paddle3D)                                            |
| :--------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------: |
| <img src='https://user-images.githubusercontent.com/61035602/206095864-f174835d-4e9a-42f7-96b8-d684fc3a3687.png' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/61035602/206095892-934be83a-f869-4a31-8e52-1074184149d1.jpg' height="126px" width="180px"> |  <img src='https://user-images.githubusercontent.com/61035602/206111796-d9a9702a-c1a0-4647-b8e9-3e1307e9d34c.png' height="126px" width="180px">  | <img src='https://user-images.githubusercontent.com/61035602/206095622-cf6dbd26-5515-472f-9451-b39bbef5b1bf.gif' height="126px" width="180px"> |
|                                                              [**人脸检测**](#模型库)                                                               |                                                [**2D关键点检测**](#️pp-tinypose-人体骨骼关键点识别)                                                 |                                                  [**多目标追踪**](#pp-tracking-实时多目标跟踪系统)                                                   |                                                              [**实例分割**](#模型库)                                                               |
| <img src='https://user-images.githubusercontent.com/61035602/206095684-72f42233-c9c7-4bd8-9195-e34859bd08bf.jpg' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/61035602/206100220-ab01d347-9ff9-4f17-9718-290ec14d4205.gif' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/61035602/206111753-836e7827-968e-4c80-92ef-7a78766892fc.gif' height="126px" width="180px"  > | <img src='https://user-images.githubusercontent.com/61035602/206095831-cc439557-1a23-4a99-b6b0-b6f2e97e8c57.jpg' height="126px" width="180px"> |
|                                               [**车辆分析——车牌识别**](#️pp-vehicle-实时车辆分析工具)                                               |                                               [**车辆分析——车流统计**](#️pp-vehicle-实时车辆分析工具)                                               |                                                [**车辆分析——违章检测**](#️pp-vehicle-实时车辆分析工具)                                                |                                               [**车辆分析——属性分析**](#️pp-vehicle-实时车辆分析工具)                                               |
| <img src='https://user-images.githubusercontent.com/61035602/206099328-2a1559e0-3b48-4424-9bad-d68f9ba5ba65.gif' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/61035602/206095918-d0e7ad87-7bbb-40f1-bcc1-37844e2271ff.gif' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/61035602/206100295-7762e1ab-ffce-44fb-b69d-45fb93657fa0.gif' height="126px" width="180px"  > | <img src='https://user-images.githubusercontent.com/61035602/206095905-8255776a-d8e6-4af1-b6e9-8d9f97e5059d.gif' height="126px" width="180px"> |
|                                                [**行人分析——闯入分析**](#pp-human-实时行人分析工具)                                                |                                                [**行人分析——行为分析**](#pp-human-实时行人分析工具)                                                |                                                 [**行人分析——属性分析**](#pp-human-实时行人分析工具)                                                 |                                                [**行人分析——人流统计**](#pp-human-实时行人分析工具)                                                |
| <img src='https://user-images.githubusercontent.com/61035602/206095792-ae0ac107-cd8e-492a-8baa-32118fc82b04.gif' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/61035602/206095778-fdd73e5d-9f91-48c7-9d3d-6f2e02ec3f79.gif' height="126px" width="180px"> |  <img src='https://user-images.githubusercontent.com/61035602/206095709-2c3a209e-6626-45dd-be16-7f0bf4d48a14.gif' height="126px" width="180px">  | <img src="https://user-images.githubusercontent.com/61035602/206113351-cc59df79-8672-4d76-b521-a15acf69ae78.gif" height="126px" width="180px"> |

同时，PaddleDetection提供了模型的在线体验功能，用户可以选择自己的数据进行在线推理。

`说明`：考虑到服务器负载压力，在线推理均为CPU推理，完整的模型开发实例以及产业部署实践代码示例请前往[🎗️产业特色模型|产业工具](#️产业特色模型产业工具-1)。

`传送门`：[模型在线体验](https://www.paddlepaddle.org.cn/models)

<div align="center">
<p align="center">
  <img src="https://user-images.githubusercontent.com/61035602/206896755-bd0cd498-1149-4e94-ae30-da590ea78a7a.gif" align="middle"/>
</p>
</div>

## 📣最新进展

**🔥超越YOLOv8，飞桨推出精度最高的实时检测器RT-DETR！**

  <div align="center">
  <img src="https://github.com/PaddlePaddle/PaddleDetection/assets/17582080/196b0a10-d2e8-401c-9132-54b9126e0a33"  height = "300" caption='' />
  <p></p>
  </div>

  - `RT-DETR解读文章传送门`：
    -  [《超越YOLOv8，飞桨推出精度最高的实时检测器RT-DETR！》](https://mp.weixin.qq.com/s/o03QM2rZNjHVto36gcV0Yw)
  - `代码传送门`：[RT-DETR](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/rtdetr)

## 👫开源社区

- **📑项目合作：** 如果您是企业开发者且有明确的目标检测垂类应用需求，请扫描如下二维码入群，并联系`群管理员AI`后可免费与官方团队展开不同层次的合作。
- **🏅️社区贡献：** PaddleDetection非常欢迎你加入到飞桨社区的开源建设中，参与贡献方式可以参考[开源项目开发指南](docs/contribution/README.md)。
- **💻直播教程：** PaddleDetection会定期在飞桨直播间([B站:飞桨PaddlePaddle](https://space.bilibili.com/476867757)、[微信: 飞桨PaddlePaddle](https://mp.weixin.qq.com/s/6ji89VKqoXDY6SSGkxS8NQ))，针对发新内容、以及产业范例、使用教程等进行直播分享。
- **🎁加入社区：** **微信扫描二维码并填写问卷之后，可以及时获取如下信息，包括：**
  - 社区最新文章、直播课等活动预告
  - 往期直播录播&PPT
  - 30+行人车辆等垂类高性能预训练模型
  - 七大任务开源数据集下载链接汇总
  - 40+前沿检测领域顶会算法
  - 15+从零上手目标检测理论与实践视频课程
  - 10+工业安防交通全流程项目实操（含源码）

<div align="center">
<img src="https://github.com/PaddlePaddle/PaddleDetection/assets/22989727/0466954b-ab4d-4984-bd36-796c37f0ee9c"  width = "150" height = "150",caption='' />
<p>PaddleDetection官方交流群二维码</p>
</div>

- **🎈社区近期活动**
  - **🔥PaddleDetection v2.6版本更新解读**

    <div align="center">
    <img src="https://user-images.githubusercontent.com/61035602/224244188-da8495fc-eea9-432f-bc2d-6f0144c2dde9.png"  height = "250" caption='' />
    <p></p>
    </div>

    - `v2.6版本版本更新解读文章传送门`：[《PaddleDetection v2.6发布：目标小？数据缺？标注累？泛化差？PP新员逐一应对！》](https://mp.weixin.qq.com/s/SLITj5k120d_fQc7jEO8Vw)

  - **🏆半监督检测**

    - `文章传送门`：[CVPR 2023 | 单阶段半监督目标检测SOTA：ARSL](https://mp.weixin.qq.com/s/UZLIGL6va2KBfofC-nKG4g)
    - `代码传送门`：[ARSL](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/semi_det)

    <div align="center">
    <img src="https://user-images.githubusercontent.com/61035602/230522850-21873665-ba79-4f8d-8dce-43d736111df8.png"  height = "250" caption='' />
    <p></p>
    </div>

  - **👀YOLO系列专题**

    - `文章传送门`：[YOLOv8来啦！YOLO内卷期模型怎么选？9+款AI硬件如何快速部署？深度解析](https://mp.weixin.qq.com/s/rPwprZeHEpmGOe5wxrmO5g)
    - `代码传送门`：[PaddleYOLO全系列](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/docs/feature_models/PaddleYOLO_MODEL.md)

    <div align="center">
    <img src="https://user-images.githubusercontent.com/61035602/213202797-3a1b24f3-53c0-4094-bb31-db2f84438fbc.jpeg"  height = "250" caption='' />
    <p></p>
    </div>

  - **🎯少目标迁移学习专题**
    - `文章传送门`：[囿于数据少？泛化性差？PaddleDetection少样本迁移学习助你一键突围！](https://mp.weixin.qq.com/s/dFEQoxSzVCOaWVZPb3N7WA)

  - **⚽️2022卡塔尔世界杯专题**
    - `文章传送门`：[世界杯决赛号角吹响！趁周末来搭一套足球3D+AI量化分析系统吧！](https://mp.weixin.qq.com/s/koJxjWDPBOlqgI-98UsfKQ)

    <div align="center">
    <img src="https://user-images.githubusercontent.com/61035602/208036574-f151a7ff-a5f1-4495-9316-a47218a6576b.gif"  height = "250" caption='' />
    <p></p>
    </div>

  - **🔍旋转框小目标检测专题**
    - `文章传送门`：[Yes, PP-YOLOE！80.73mAP、38.5mAP，旋转框、小目标检测能力双SOTA！](https://mp.weixin.qq.com/s/6ji89VKqoXDY6SSGkxS8NQ)

    <div align="center">
    <img src="https://user-images.githubusercontent.com/61035602/208037368-5b9f01f7-afd9-46d8-bc80-271ccb5db7bb.png"  height = "220" caption='' />
    <p></p>
    </div>

  - **🎊YOLO Vision世界学术交流大会**
    - **PaddleDetection**受邀参与首个以**YOLO为主题**的**YOLO-VISION**世界大会，与全球AI领先开发者学习交流。
    - `活动链接传送门`：[YOLO-VISION](https://ultralytics.com/yolo-vision)

    <div  align="center">
    <img src="https://user-images.githubusercontent.com/48054808/192301374-940cf2fa-9661-419b-9c46-18a4570df381.jpeg" width="400"/>
    </div>

- **🏅️社区贡献**
  - `活动链接传送门`：[Yes, PP-YOLOE! 基于PP-YOLOE的算法开发](https://github.com/PaddlePaddle/PaddleDetection/issues/7345)

## ✨主要特性

#### 🧩模块化设计
PaddleDetection将检测模型解耦成不同的模块组件，通过自定义模块组件组合，用户可以便捷高效地完成检测模型的搭建。`传送门`：[🧩模块组件](#模块组件)。

#### 📱丰富的模型库
PaddleDetection支持大量的最新主流的算法基准以及预训练模型，涵盖2D/3D目标检测、实例分割、人脸检测、关键点检测、多目标跟踪、半监督学习等方向。`传送门`：[📱模型库](#模型库)、[⚖️模型性能对比](#️模型性能对比)。

#### 🎗️产业特色模型|产业工具
PaddleDetection打造产业级特色模型以及分析工具：PP-YOLOE+、PP-PicoDet、PP-TinyPose、PP-HumanV2、PP-Vehicle等，针对通用、高频垂类应用场景提供深度优化解决方案以及高度集成的分析工具，降低开发者的试错、选择成本，针对业务场景快速应用落地。`传送门`：[🎗️产业特色模型|产业工具](#️产业特色模型产业工具-1)。

#### 💡🏆产业级部署实践
PaddleDetection整理工业、农业、林业、交通、医疗、金融、能源电力等AI应用范例，打通数据标注-模型训练-模型调优-预测部署全流程，持续降低目标检测技术产业落地门槛。`传送门`：[💡产业实践范例](#产业实践范例)、[🏆企业应用案例](#企业应用案例)。

<div align="center">
<p align="center">
  <img src="https://user-images.githubusercontent.com/61035602/206431371-912a14c8-ce1e-48ec-ae6f-7267016b308e.png" align="middle" width="1280"/>
</p>
</div>


## 🍱安装

参考[安装说明](docs/tutorials/INSTALL_cn.md)进行安装。

## 🔥教程

**深度学习入门教程**

- [零基础入门深度学习](https://www.paddlepaddle.org.cn/tutorials/projectdetail/4676538)
- [零基础入门目标检测](https://aistudio.baidu.com/aistudio/education/group/info/1617)

**快速开始**

- [快速体验](docs/tutorials/QUICK_STARTED_cn.md)
- [示例：30分钟快速开发交通标志检测模型](docs/tutorials/GETTING_STARTED_cn.md)

**数据准备**
- [数据准备](docs/tutorials/data/README.md)
- [数据处理模块](docs/advanced_tutorials/READER.md)

**配置文件说明**
- [RCNN参数说明](docs/tutorials/config_annotation/faster_rcnn_r50_fpn_1x_coco_annotation.md)
- [PP-YOLO参数说明](docs/tutorials/config_annotation/ppyolo_r50vd_dcn_1x_coco_annotation.md)

**模型开发**

- [新增检测模型](docs/advanced_tutorials/MODEL_TECHNICAL.md)
- 二次开发
  - [目标检测](docs/advanced_tutorials/customization/detection.md)
  - [关键点检测](docs/advanced_tutorials/customization/keypoint_detection.md)
  - [多目标跟踪](docs/advanced_tutorials/customization/pphuman_mot.md)
  - [行为识别](docs/advanced_tutorials/customization/action_recognotion/)
  - [属性识别](docs/advanced_tutorials/customization/pphuman_attribute.md)

**部署推理**

- [模型导出教程](deploy/EXPORT_MODEL.md)
- [模型压缩](https://github.com/PaddlePaddle/PaddleSlim)
  - [剪裁/量化/蒸馏教程](configs/slim)
- [Paddle Inference部署](deploy/README.md)
  - [Python端推理部署](deploy/python)
  - [C++端推理部署](deploy/cpp)
- [Paddle Lite部署](deploy/lite)
- [Paddle Serving部署](deploy/serving)
- [ONNX模型导出](deploy/EXPORT_ONNX_MODEL.md)
- [推理benchmark](deploy/BENCHMARK_INFER.md)

## 🔑FAQ
- [FAQ/常见问题汇总](docs/tutorials/FAQ)

## 🧩模块组件

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
          <li><a href="ppdet/modeling/backbones/resnet.py">ResNet</a></li>
          <li><a href="ppdet/modeling/backbones/res2net.py">CSPResNet</a></li>
          <li><a href="ppdet/modeling/backbones/senet.py">SENet</a></li>
          <li><a href="ppdet/modeling/backbones/res2net.py">Res2Net</a></li>
          <li><a href="ppdet/modeling/backbones/hrnet.py">HRNet</a></li>
          <li><a href="ppdet/modeling/backbones/lite_hrnet.py">Lite-HRNet</a></li>
          <li><a href="ppdet/modeling/backbones/darknet.py">DarkNet</a></li>
          <li><a href="ppdet/modeling/backbones/csp_darknet.py">CSPDarkNet</a></li>
          <li><a href="ppdet/modeling/backbones/mobilenet_v1.py">MobileNetV1</a></li>
          <li><a href="ppdet/modeling/backbones/mobilenet_v3.py">MobileNetV1</a></li>  
          <li><a href="ppdet/modeling/backbones/shufflenet_v2.py">ShuffleNetV2</a></li>
          <li><a href="ppdet/modeling/backbones/ghostnet.py">GhostNet</a></li>
          <li><a href="ppdet/modeling/backbones/blazenet.py">BlazeNet</a></li>
          <li><a href="ppdet/modeling/backbones/dla.py">DLA</a></li>
          <li><a href="ppdet/modeling/backbones/hardnet.py">HardNet</a></li>
          <li><a href="ppdet/modeling/backbones/lcnet.py">LCNet</a></li>  
          <li><a href="ppdet/modeling/backbones/esnet.py">ESNet</a></li>  
          <li><a href="ppdet/modeling/backbones/swin_transformer.py">Swin-Transformer</a></li>
          <li><a href="ppdet/modeling/backbones/convnext.py">ConvNeXt</a></li>
          <li><a href="ppdet/modeling/backbones/vgg.py">VGG</a></li>
          <li><a href="ppdet/modeling/backbones/vision_transformer.py">Vision Transformer</a></li>
          <li><a href="configs/convnext">ConvNext</a></li>
      </ul>
      </td>
      <td>
      <ul>
        <li><a href="ppdet/modeling/necks/bifpn.py">BiFPN</a></li>
        <li><a href="ppdet/modeling/necks/blazeface_fpn.py">BlazeFace-FPN</a></li>
        <li><a href="ppdet/modeling/necks/centernet_fpn.py">CenterNet-FPN</a></li>
        <li><a href="ppdet/modeling/necks/csp_pan.py">CSP-PAN</a></li>
        <li><a href="ppdet/modeling/necks/custom_pan.py">Custom-PAN</a></li>
        <li><a href="ppdet/modeling/necks/fpn.py">FPN</a></li>
        <li><a href="ppdet/modeling/necks/es_pan.py">ES-PAN</a></li>
        <li><a href="ppdet/modeling/necks/hrfpn.py">HRFPN</a></li>
        <li><a href="ppdet/modeling/necks/lc_pan.py">LC-PAN</a></li>
        <li><a href="ppdet/modeling/necks/ttf_fpn.py">TTF-FPN</a></li>
        <li><a href="ppdet/modeling/necks/yolo_fpn.py">YOLO-FPN</a></li>
      </ul>
      </td>
      <td>
        <ul>
          <li><a href="ppdet/modeling/losses/smooth_l1_loss.py">Smooth-L1</a></li>
          <li><a href="ppdet/modeling/losses/detr_loss.py">Detr Loss</a></li>
          <li><a href="ppdet/modeling/losses/fairmot_loss.py">Fairmot Loss</a></li>
          <li><a href="ppdet/modeling/losses/fcos_loss.py">Fcos Loss</a></li>
          <li><a href="ppdet/modeling/losses/gfocal_loss.py">GFocal Loss</a></li>
          <li><a href="ppdet/modeling/losses/jde_loss.py">JDE Loss</a></li>
          <li><a href="ppdet/modeling/losses/keypoint_loss.py">KeyPoint Loss</a></li>
          <li><a href="ppdet/modeling/losses/solov2_loss.py">SoloV2 Loss</a></li>
          <li><a href="ppdet/modeling/losses/focal_loss.py">Focal Loss</a></li>
          <li><a href="ppdet/modeling/losses/iou_loss.py">GIoU/DIoU/CIoU</a></li>  
          <li><a href="ppdet/modeling/losses/iou_aware_loss.py">IoUAware</a></li>
          <li><a href="ppdet/modeling/losses/sparsercnn_loss.py">SparseRCNN Loss</a></li>
          <li><a href="ppdet/modeling/losses/ssd_loss.py">SSD Loss</a></li>
          <li><a href="ppdet/modeling/losses/focal_loss.py">YOLO Loss</a></li>
          <li><a href="ppdet/modeling/losses/yolo_loss.py">CT Focal Loss</a></li>
          <li><a href="ppdet/modeling/losses/varifocal_loss.py">VariFocal Loss</a></li>
        </ul>
      </td>
      <td>
      </ul>
          <li><b>Post-processing</b></li>
        <ul>
        <ul>
           <li><a href="ppdet/modeling/post_process.py">SoftNMS</a></li>
            <li><a href="ppdet/modeling/post_process.py">MatrixNMS</a></li>
            </ul>
            </ul>
          <li><b>Training</b></li>
        <ul>
        <ul>
            <li><a href="tools/train.py#L62">FP16 training</a></li>
            <li><a href="docs/tutorials/DistributedTraining_cn.md">Multi-machine training </a></li>
                        </ul>
            </ul>
          <li><b>Common</b></li>
        <ul>
        <ul>
            <li><a href="ppdet/modeling/backbones/resnet.py#L41">Sync-BN</a></li>
            <li><a href="configs/gn/README.md">Group Norm</a></li>
            <li><a href="configs/dcn/README.md">DCNv2</a></li>
            <li><a href="ppdet/optimizer/ema.py">EMA</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="ppdet/data/transform/operators.py">Resize</a></li>  
          <li><a href="ppdet/data/transform/operators.py">Lighting</a></li>  
          <li><a href="ppdet/data/transform/operators.py">Flipping</a></li>  
          <li><a href="ppdet/data/transform/operators.py">Expand</a></li>
          <li><a href="ppdet/data/transform/operators.py">Crop</a></li>
          <li><a href="ppdet/data/transform/operators.py">Color Distort</a></li>  
          <li><a href="ppdet/data/transform/operators.py">Random Erasing</a></li>  
          <li><a href="ppdet/data/transform/operators.py">Mixup </a></li>
          <li><a href="ppdet/data/transform/operators.py">AugmentHSV</a></li>
          <li><a href="ppdet/data/transform/operators.py">Mosaic</a></li>
          <li><a href="ppdet/data/transform/operators.py">Cutmix </a></li>
          <li><a href="ppdet/data/transform/operators.py">Grid Mask</a></li>
          <li><a href="ppdet/data/transform/operators.py">Auto Augment</a></li>  
          <li><a href="ppdet/data/transform/operators.py">Random Perspective</a></li>  
        </ul>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>

## 📱模型库

<table align="center">
  <tbody>
    <tr align="center" valign="center">
      <td>
        <b>2D Detection</b>
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
            <li><a href="configs/faster_rcnn/README.md">Faster RCNN</a></li>
            <li><a href="ppdet/modeling/necks/fpn.py">FPN</a></li>
            <li><a href="configs/cascade_rcnn/README.md">Cascade-RCNN</a></li>
            <li><a href="configs/rcnn_enhance">PSS-Det</a></li>
            <li><a href="configs/retinanet/README.md">RetinaNet</a></li>
            <li><a href="configs/yolov3/README.md">YOLOv3</a></li>  
            <li><a href="configs/yolof/README.md">YOLOF</a></li>  
            <li><a href="configs/yolox/README.md">YOLOX</a></li>  
            <li><a href="https://github.com/PaddlePaddle/PaddleYOLO/tree/develop/configs/yolov5">YOLOv5</a></li>
            <li><a href="https://github.com/PaddlePaddle/PaddleYOLO/tree/develop/configs/yolov6">YOLOv6</a></li>  
            <li><a href="https://github.com/PaddlePaddle/PaddleYOLO/tree/develop/configs/yolov7">YOLOv7</a></li>
            <li><a href="https://github.com/PaddlePaddle/PaddleYOLO/tree/develop/configs/yolov8">YOLOv8</a></li>
            <li><a href="https://github.com/PaddlePaddle/PaddleYOLO/tree/develop/configs/rtmdet">RTMDet</a></li>
            <li><a href="configs/ppyolo/README_cn.md">PP-YOLO</a></li>
            <li><a href="configs/ppyolo#pp-yolo-tiny">PP-YOLO-Tiny</a></li>
            <li><a href="configs/picodet">PP-PicoDet</a></li>
            <li><a href="configs/ppyolo/README_cn.md">PP-YOLOv2</a></li>
            <li><a href="configs/ppyoloe/README_legacy.md">PP-YOLOE</a></li>
            <li><a href="configs/ppyoloe/README_cn.md">PP-YOLOE+</a></li>
            <li><a href="configs/smalldet">PP-YOLOE-SOD</a></li>
            <li><a href="configs/rotate/README.md">PP-YOLOE-R</a></li>
            <li><a href="configs/ssd/README.md">SSD</a></li>
            <li><a href="configs/centernet">CenterNet</a></li>
            <li><a href="configs/fcos">FCOS</a></li>  
            <li><a href="configs/rotate/fcosr">FCOSR</a></li>  
            <li><a href="configs/ttfnet">TTFNet</a></li>
            <li><a href="configs/tood">TOOD</a></li>
            <li><a href="configs/gfl">GFL</a></li>
            <li><a href="configs/gfl/gflv2_r50_fpn_1x_coco.yml">GFLv2</a></li>
            <li><a href="configs/detr">DETR</a></li>
            <li><a href="configs/deformable_detr">Deformable DETR</a></li>
            <li><a href="configs/sparse_rcnn">Sparse RCNN</a></li>
      </ul>
      </td>
      <td>
        <ul>
           <li><a href="configs/mot/jde">JDE</a></li>
            <li><a href="configs/mot/fairmot">FairMOT</a></li>
            <li><a href="configs/mot/deepsort">DeepSORT</a></li>
            <li><a href="configs/mot/bytetrack">ByteTrack</a></li>
            <li><a href="configs/mot/ocsort">OC-SORT</a></li>
            <li><a href="configs/mot/botsort">BoT-SORT</a></li>
            <li><a href="configs/mot/centertrack">CenterTrack</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="configs/keypoint/hrnet">HRNet</a></li>
            <li><a href="configs/keypoint/higherhrnet">HigherHRNet</a></li>
            <li><a href="configs/keypoint/lite_hrnet">Lite-HRNet</a></li>
            <li><a href="configs/keypoint/tiny_pose">PP-TinyPose</a></li>
        </ul>
</td>
<td>
</ul>
          <li><b>Instance Segmentation</b></li>
        <ul>
        <ul>
          <li><a href="configs/mask_rcnn">Mask RCNN</a></li>
            <li><a href="configs/cascade_rcnn">Cascade Mask RCNN</a></li>
            <li><a href="configs/solov2">SOLOv2</a></li>
        </ul>
      </ul>
          <li><b>Face Detection</b></li>
        <ul>
        <ul>
            <li><a href="configs/face_detection">BlazeFace</a></li>
        </ul>
        </ul>
          <li><b>Semi-Supervised Detection</b></li>
        <ul>
        <ul>
            <li><a href="configs/semi_det">DenseTeacher</a></li>
        </ul>
        </ul>
          <li><b>3D Detection</b></li>
        <ul>
        <ul>
            <li><a href="https://github.com/PaddlePaddle/Paddle3D">Smoke</a></li>
            <li><a href="https://github.com/PaddlePaddle/Paddle3D">CaDDN</a></li>
            <li><a href="https://github.com/PaddlePaddle/Paddle3D">PointPillars</a></li>
            <li><a href="https://github.com/PaddlePaddle/Paddle3D">CenterPoint</a></li>
            <li><a href="https://github.com/PaddlePaddle/Paddle3D">SequeezeSegV3</a></li>
            <li><a href="https://github.com/PaddlePaddle/Paddle3D">IA-SSD</a></li>
            <li><a href="https://github.com/PaddlePaddle/Paddle3D">PETR</a></li>
        </ul>
        </ul>
          <li><b>Vehicle Analysis Toolbox</b></li>
        <ul>
        <ul>
            <li><a href="deploy/pipeline/README.md">PP-Vehicle</a></li>
        </ul>
        </ul>
          <li><b>Human Analysis Toolbox</b></li>
        <ul>
        <ul>
            <li><a href="deploy/pipeline/README.md">PP-Human</a></li>
            <li><a href="deploy/pipeline/README.md">PP-HumanV2</a></li>
        </ul>
        </ul>
          <li><b>Sport Analysis Toolbox</b></li>
        <ul>
        <ul>
            <li><a href="https://github.com/PaddlePaddle/PaddleSports">PP-Sports</a></li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

## ⚖️模型性能对比

#### 🖥️服务器端模型性能对比

各模型结构和骨干网络的代表模型在COCO数据集上精度mAP和单卡Tesla V100上预测速度(FPS)对比图。

  <div  align="center">
  <img src="https://user-images.githubusercontent.com/61035602/206434766-caaa781b-b922-481f-af09-15faac9ed33b.png" width="800"/>
</div>

<details>
<summary><b> 测试说明(点击展开)</b></summary>

- ViT为ViT-Cascade-Faster-RCNN模型，COCO数据集mAP高达55.7%
- Cascade-Faster-RCNN为Cascade-Faster-RCNN-ResNet50vd-DCN，PaddleDetection将其优化到COCO数据mAP为47.8%时推理速度为20FPS
- PP-YOLOE是对PP-YOLO v2模型的进一步优化，L版本在COCO数据集mAP为51.6%，Tesla V100预测速度78.1FPS
- PP-YOLOE+是对PPOLOE模型的进一步优化，L版本在COCO数据集mAP为53.3%，Tesla V100预测速度78.1FPS
- YOLOX和YOLOv5均为基于PaddleDetection复现算法，YOLOv5代码在[PaddleYOLO](https://github.com/PaddlePaddle/PaddleYOLO)中，参照[PaddleYOLO_MODEL](docs/feature_models/PaddleYOLO_MODEL.md)
- 图中模型均可在[📱模型库](#模型库)中获取
</details>

#### ⌚️移动端模型性能对比

各移动端模型在COCO数据集上精度mAP和高通骁龙865处理器上预测速度(FPS)对比图。

  <div  align="center">
  <img src="https://user-images.githubusercontent.com/61035602/206434741-10460690-8fc3-4084-a11a-16fe4ce2fc85.png" width="550"/>
</div>


<details>
<summary><b> 测试说明(点击展开)</b></summary>

- 测试数据均使用高通骁龙865(4xA77+4xA55)处理器，batch size为1, 开启4线程测试，测试使用NCNN预测库，测试脚本见[MobileDetBenchmark](https://github.com/JiweiMaster/MobileDetBenchmark)
- PP-PicoDet及PP-YOLO-Tiny为PaddleDetection自研模型，可在[📱模型库](#模型库)中获取，其余模型PaddleDetection暂未提供
</details>

## 🎗️产业特色模型|产业工具

产业特色模型｜产业工具是PaddleDetection针对产业高频应用场景打造的兼顾精度和速度的模型以及工具箱，注重从数据处理-模型训练-模型调优-模型部署的端到端打通，且提供了实际生产环境中的实践范例代码，帮助拥有类似需求的开发者高效的完成产品开发落地应用。

该系列模型｜工具均已PP前缀命名，具体介绍、预训练模型以及产业实践范例代码如下。

### 💎PP-YOLOE 高精度目标检测模型

<details>
<summary><b> 简介(点击展开)</b></summary>

PP-YOLOE是基于PP-YOLOv2的卓越的单阶段Anchor-free模型，超越了多种流行的YOLO模型。PP-YOLOE避免了使用诸如Deformable Convolution或者Matrix NMS之类的特殊算子，以使其能轻松地部署在多种多样的硬件上。其使用大规模数据集obj365预训练模型进行预训练，可以在不同场景数据集上快速调优收敛。

`传送门`：[PP-YOLOE说明](configs/ppyoloe/README_cn.md)。

`传送门`：[arXiv论文](https://arxiv.org/abs/2203.16250)。

</details>

<details>
<summary><b> 预训练模型(点击展开)</b></summary>

| 模型名称    | COCO精度（mAP） | V100 TensorRT FP16速度(FPS) | 推荐部署硬件 |                        配置文件                         |                                        模型下载                                         |
| :---------- | :-------------: | :-------------------------: | :----------: | :-----------------------------------------------------: | :-------------------------------------------------------------------------------------: |
| PP-YOLOE+_l |      53.3       |            149.2            |    服务器    | [链接](configs/ppyoloe/ppyoloe_plus_crn_l_80e_coco.yml) | [下载地址](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_m_80e_coco.pdparams) |

`传送门`：[全部预训练模型](configs/ppyoloe/README_cn.md)。
</details>

<details>
<summary><b> 产业应用代码示例(点击展开)</b></summary>

| 行业 | 类别              | 亮点                                                                                          | 文档说明                                                      | 模型下载                                            |
| ---- | ----------------- | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------- | --------------------------------------------------- |
| 农业 | 农作物检测        | 用于葡萄栽培中基于图像的监测和现场机器人技术，提供了来自5种不同葡萄品种的实地实例             | [PP-YOLOE+ 下游任务](./configs/ppyoloe/application/README.md) | [下载链接](./configs/ppyoloe/application/README.md) |
| 通用 | 低光场景检测      | 低光数据集使用ExDark，包括从极低光环境到暮光环境等10种不同光照条件下的图片。                  | [PP-YOLOE+ 下游任务](./configs/ppyoloe/application/README.md) | [下载链接](./configs/ppyoloe/application/README.md) |
| 工业 | PCB电路板瑕疵检测 | 工业数据集使用PKU-Market-PCB，该数据集用于印刷电路板（PCB）的瑕疵检测，提供了6种常见的PCB缺陷 | [PP-YOLOE+ 下游任务](./configs/ppyoloe/application/README.md) | [下载链接](./configs/ppyoloe/application/README.md) |
</details>

### 💎PP-YOLOE-R 高性能旋转框检测模型

<details>
<summary><b> 简介(点击展开)</b></summary>

PP-YOLOE-R是一个高效的单阶段Anchor-free旋转框检测模型，基于PP-YOLOE+引入了一系列改进策略来提升检测精度。根据不同的硬件对精度和速度的要求，PP-YOLOE-R包含s/m/l/x四个尺寸的模型。在DOTA 1.0数据集上，PP-YOLOE-R-l和PP-YOLOE-R-x在单尺度训练和测试的情况下分别达到了78.14mAP和78.28 mAP，这在单尺度评估下超越了几乎所有的旋转框检测模型。通过多尺度训练和测试，PP-YOLOE-R-l和PP-YOLOE-R-x的检测精度进一步提升至80.02mAP和80.73 mAP，超越了所有的Anchor-free方法并且和最先进的Anchor-based的两阶段模型精度几乎相当。在保持高精度的同时，PP-YOLOE-R避免使用特殊的算子，例如Deformable Convolution或Rotated RoI Align，使其能轻松地部署在多种多样的硬件上。

`传送门`：[PP-YOLOE-R说明](configs/rotate/ppyoloe_r)。

`传送门`：[arXiv论文](https://arxiv.org/abs/2211.02386)。

</details>

<details>
<summary><b> 预训练模型(点击展开)</b></summary>

|     模型     | Backbone |  mAP  | V100 TRT FP16 (FPS) | RTX 2080 Ti TRT FP16 (FPS) | Params (M) | FLOPs (G) | 学习率策略 | 角度表示 | 数据增广 | GPU数目 | 每GPU图片数目 |                                      模型下载                                       |                                                            配置文件                                                            |
| :----------: | :------: | :---: | :-----------------: | :------------------------: | :--------: | :-------: | :--------: | :------: | :------: | :-----: | :-----------: | :---------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------: |
| PP-YOLOE-R-l |  CRN-l   | 80.02 |        69.7         |            48.3            |   53.29    |  281.65   |     3x     |    oc    |  MS+RR   |    4    |       2       | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_r_crn_l_3x_dota_ms.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/rotate/ppyoloe_r/ppyoloe_r_crn_l_3x_dota_ms.yml) |

`传送门`：[全部预训练模型](configs/rotate/ppyoloe_r)。
</details>

<details>
<summary><b> 产业应用代码示例(点击展开)</b></summary>

| 行业 | 类别       | 亮点                                                                  | 文档说明                                                                                | 模型下载                                                              |
| ---- | ---------- | --------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| 通用 | 旋转框检测 | 手把手教你上手PP-YOLOE-R旋转框检测，10分钟将脊柱数据集精度训练至95mAP | [基于PP-YOLOE-R的旋转框检测](https://aistudio.baidu.com/aistudio/projectdetail/5058293) | [下载链接](https://aistudio.baidu.com/aistudio/projectdetail/5058293) |
</details>

### 💎PP-YOLOE-SOD 高精度小目标检测模型

<details>
<summary><b> 简介(点击展开)</b></summary>

PP-YOLOE-SOD(Small Object Detection)是PaddleDetection团队针对小目标检测提出的检测方案，在VisDrone-DET数据集上单模型精度达到38.5mAP，达到了SOTA性能。其分别基于切图拼图流程优化的小目标检测方案以及基于原图模型算法优化的小目标检测方案。同时提供了数据集自动分析脚本，只需输入数据集标注文件，便可得到数据集统计结果，辅助判断数据集是否是小目标数据集以及是否需要采用切图策略，同时给出网络超参数参考值。

`传送门`：[PP-YOLOE-SOD 小目标检测模型](configs/smalldet)。

</details>

<details>
<summary><b> 预训练模型(点击展开)</b></summary>
- VisDrone数据集预训练模型

| 模型                | COCOAPI mAP<sup>val<br>0.5:0.95 | COCOAPI mAP<sup>val<br>0.5 | COCOAPI mAP<sup>test_dev<br>0.5:0.95 | COCOAPI mAP<sup>test_dev<br>0.5 | MatlabAPI mAP<sup>test_dev<br>0.5:0.95 | MatlabAPI mAP<sup>test_dev<br>0.5 |                                              下载                                               |                           配置文件                           |
| :------------------ | :-----------------------------: | :------------------------: | :----------------------------------: | :-----------------------------: | :------------------------------------: | :-------------------------------: | :---------------------------------------------------------------------------------------------: | :----------------------------------------------------------: |
| **PP-YOLOE+_SOD-l** |            **31.9**             |          **52.1**          |               **25.6**               |            **43.5**             |               **30.25**                |             **51.18**             | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_sod_crn_l_80e_visdrone.pdparams) | [配置文件](visdrone/ppyoloe_plus_sod_crn_l_80e_visdrone.yml) |

`传送门`：[全部预训练模型](configs/smalldet)。
</details>

<details>
<summary><b> 产业应用代码示例(点击展开)</b></summary>

| 行业 | 类别       | 亮点                                                 | 文档说明                                                                                          | 模型下载                                                              |
| ---- | ---------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| 通用 | 小目标检测 | 基于PP-YOLOE-SOD的无人机航拍图像检测案例全流程实操。 | [基于PP-YOLOE-SOD的无人机航拍图像检测](https://aistudio.baidu.com/aistudio/projectdetail/5036782) | [下载链接](https://aistudio.baidu.com/aistudio/projectdetail/5036782) |
</details>

### 💫PP-PicoDet 超轻量实时目标检测模型

<details>
<summary><b> 简介(点击展开)</b></summary>

全新的轻量级系列模型PP-PicoDet，在移动端具有卓越的性能，成为全新SOTA轻量级模型。

`传送门`：[PP-PicoDet说明](configs/picodet/README.md)。

`传送门`：[arXiv论文](https://arxiv.org/abs/2111.00902)。

</details>

<details>
<summary><b> 预训练模型(点击展开)</b></summary>

| 模型名称  | COCO精度（mAP） | 骁龙865 四线程速度(FPS) |  推荐部署硬件  |                       配置文件                       |                                       模型下载                                       |
| :-------- | :-------------: | :---------------------: | :------------: | :--------------------------------------------------: | :----------------------------------------------------------------------------------: |
| PicoDet-L |      36.1       |          39.7           | 移动端、嵌入式 | [链接](configs/picodet/picodet_l_320_coco_lcnet.yml) | [下载地址](https://paddledet.bj.bcebos.com/models/picodet_l_320_coco_lcnet.pdparams) |

`传送门`：[全部预训练模型](configs/picodet/README.md)。
</details>


<details>
<summary><b> 产业应用代码示例(点击展开)</b></summary>

| 行业     | 类别         | 亮点                                                                                                                           | 文档说明                                                                                                          | 模型下载                                                                                      |
| -------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| 智慧城市 | 道路垃圾检测 | 通过在市政环卫车辆上安装摄像头对路面垃圾检测并分析，实现对路面遗撒的垃圾进行监控，记录并通知环卫人员清理，大大提升了环卫人效。 | [基于PP-PicoDet的路面垃圾检测](https://aistudio.baidu.com/aistudio/projectdetail/3846170?channelType=0&channel=0) | [下载链接](https://aistudio.baidu.com/aistudio/projectdetail/3846170?channelType=0&channel=0) |
</details>

### 📡PP-Tracking 实时多目标跟踪系统

<details>
<summary><b> 简介(点击展开)</b></summary>

PaddleDetection团队提供了实时多目标跟踪系统PP-Tracking，是基于PaddlePaddle深度学习框架的业界首个开源的实时多目标跟踪系统，具有模型丰富、应用广泛和部署高效三大优势。 PP-Tracking支持单镜头跟踪(MOT)和跨镜头跟踪(MTMCT)两种模式，针对实际业务的难点和痛点，提供了行人跟踪、车辆跟踪、多类别跟踪、小目标跟踪、流量统计以及跨镜头跟踪等各种多目标跟踪功能和应用，部署方式支持API调用和GUI可视化界面，部署语言支持Python和C++，部署平台环境支持Linux、NVIDIA Jetson等。

`传送门`：[PP-Tracking说明](configs/mot/README.md)。

</details>

<details>
<summary><b> 预训练模型(点击展开)</b></summary>

| 模型名称  |               模型简介               |          精度          | 速度(FPS) |      推荐部署硬件      |                          配置文件                          |                                              模型下载                                              |
| :-------- | :----------------------------------: | :--------------------: | :-------: | :--------------------: | :--------------------------------------------------------: | :------------------------------------------------------------------------------------------------: |
| ByteTrack |   SDE多目标跟踪算法 仅包含检测模型   |   MOT-17 test:  78.4   |     -     | 服务器、移动端、嵌入式 |     [链接](configs/mot/bytetrack/bytetrack_yolox.yml)      |  [下载地址](https://bj.bcebos.com/v1/paddledet/models/mot/yolox_x_24e_800x1440_mix_det.pdparams)   |
| FairMOT   | JDE多目标跟踪算法 多任务联合学习方法 |   MOT-16 test: 75.0    |     -     | 服务器、移动端、嵌入式 | [链接](configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml) |     [下载地址](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608.pdparams)     |
| OC-SORT   |   SDE多目标跟踪算法 仅包含检测模型   | MOT-17 half val:  75.5 |     -     | 服务器、移动端、嵌入式 |        [链接](configs/mot/ocsort/ocsort_yolox.yml)         | [下载地址](https://bj.bcebos.com/v1/paddledet/models/mot/yolox_x_24e_800x1440_mix_mot_ch.pdparams) |
</details>

<details>
<summary><b> 产业应用代码示例(点击展开)</b></summary>

| 行业 | 类别       | 亮点                       | 文档说明                                                                                       | 模型下载                                                              |
| ---- | ---------- | -------------------------- | ---------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| 通用 | 多目标跟踪 | 快速上手单镜头、多镜头跟踪 | [PP-Tracking之手把手玩转多目标跟踪](https://aistudio.baidu.com/aistudio/projectdetail/3022582) | [下载链接](https://aistudio.baidu.com/aistudio/projectdetail/3022582) |
</details>

### ⛷️PP-TinyPose 人体骨骼关键点识别

<details>
<summary><b> 简介(点击展开)</b></summary>

PaddleDetection 中的关键点检测部分紧跟最先进的算法，包括 Top-Down 和 Bottom-Up 两种方法，可以满足用户的不同需求。同时，PaddleDetection 提供针对移动端设备优化的自研实时关键点检测模型 PP-TinyPose。

`传送门`：[PP-TinyPose说明](configs/keypoint/tiny_pose)。

</details>

<details>
<summary><b> 预训练模型(点击展开)</b></summary>

|  模型名称   |               模型简介               | COCO精度（AP） |         速度(FPS)         |  推荐部署硬件  |                        配置文件                         |                                         模型下载                                         |
| :---------: | :----------------------------------: | :------------: | :-----------------------: | :------------: | :-----------------------------------------------------: | :--------------------------------------------------------------------------------------: |
| PP-TinyPose | 轻量级关键点算法<br/>输入尺寸256x192 |      68.8      | 骁龙865 四线程: 158.7 FPS | 移动端、嵌入式 | [链接](configs/keypoint/tiny_pose/tinypose_256x192.yml) | [下载地址](https://bj.bcebos.com/v1/paddledet/models/keypoint/tinypose_256x192.pdparams) |

`传送门`：[全部预训练模型](configs/keypoint/README.md)。
</details>

<details>
<summary><b> 产业应用代码示例(点击展开)</b></summary>

| 行业 | 类别 | 亮点                                                                                                                                     | 文档说明                                                                                             | 模型下载                                                              |
| ---- | ---- | ---------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| 运动 | 健身 | 提供从模型选型、数据准备、模型训练优化，到后处理逻辑和模型部署的全流程可复用方案，有效解决了复杂健身动作的高效识别，打造AI虚拟健身教练！ | [基于PP-TinyPose增强版的智能健身动作识别](https://aistudio.baidu.com/aistudio/projectdetail/4385813) | [下载链接](https://aistudio.baidu.com/aistudio/projectdetail/4385813) |
</details>

### 🏃🏻PP-Human 实时行人分析工具

<details>
<summary><b> 简介(点击展开)</b></summary>

PaddleDetection深入探索核心行业的高频场景，提供了行人开箱即用分析工具，支持图片/单镜头视频/多镜头视频/在线视频流多种输入方式，广泛应用于智慧交通、智慧城市、工业巡检等领域。支持服务器端部署及TensorRT加速，T4服务器上可达到实时。
PP-Human支持四大产业级功能：五大异常行为识别、26种人体属性分析、实时人流计数、跨镜头（ReID）跟踪。

`传送门`：[PP-Human行人分析工具使用指南](deploy/pipeline/README.md)。

</details>

<details>
<summary><b> 预训练模型(点击展开)</b></summary>

|        任务        | T4 TensorRT FP16: 速度（FPS） | 推荐部署硬件 |                                                                                                                                         模型下载                                                                                                                                         |                             模型体积                              |
| :----------------: | :---------------------------: | :----------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------: |
| 行人检测（高精度） |             39.8              |    服务器    |                                                                                              [目标检测](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)                                                                                               |                               182M                                |
| 行人跟踪（高精度） |             31.4              |    服务器    |                                                                                             [多目标跟踪](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)                                                                                              |                               182M                                |
| 属性识别（高精度） |          单人 117.6           |    服务器    |                                      [目标检测](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)<br> [属性识别](https://bj.bcebos.com/v1/paddledet/models/pipeline/PPHGNet_small_person_attribute_954_infer.zip)                                       |                  目标检测：182M<br>属性识别：86M                  |
|      摔倒识别      |           单人 100            |    服务器    | [多目标跟踪](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip) <br> [关键点检测](https://bj.bcebos.com/v1/paddledet/models/pipeline/dark_hrnet_w32_256x192.zip) <br> [基于关键点行为识别](https://bj.bcebos.com/v1/paddledet/models/pipeline/STGCN.zip) | 多目标跟踪：182M<br>关键点检测：101M<br>基于关键点行为识别：21.8M |
|      闯入识别      |             31.4              |    服务器    |                                                                                             [多目标跟踪](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)                                                                                              |                               182M                                |
|      打架识别      |             50.8              |    服务器    |                                                                                              [视频分类](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)                                                                                               |                                90M                                |
|      抽烟识别      |             340.1             |    服务器    |                                    [目标检测](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)<br>[基于人体id的目标检测](https://bj.bcebos.com/v1/paddledet/models/pipeline/ppyoloe_crn_s_80e_smoking_visdrone.zip)                                    |            目标检测：182M<br>基于人体id的目标检测：27M            |
|     打电话识别     |             166.7             |    服务器    |                                      [目标检测](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)<br>[基于人体id的图像分类](https://bj.bcebos.com/v1/paddledet/models/pipeline/PPHGNet_tiny_calling_halfbody.zip)                                       |            目标检测：182M<br>基于人体id的图像分类：45M            |

`传送门`：[完整预训练模型](deploy/pipeline/README.md)。
</details>

<details>
<summary><b> 产业应用代码示例(点击展开)</b></summary>

| 行业     | 类别     | 亮点                                                                                                                                           | 文档说明                                                                                               | 模型下载                                                                                 |
| -------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------- |
| 智能安防 | 摔倒检测 | 飞桨行人分析PP-Human中提供的摔倒识别算法，采用了关键点+时空图卷积网络的技术，对摔倒姿势无限制、背景环境无要求。                                | [基于PP-Human v2的摔倒检测](https://aistudio.baidu.com/aistudio/projectdetail/4606001)                 | [下载链接](https://aistudio.baidu.com/aistudio/projectdetail/4606001)                    |
| 智能安防 | 打架识别 | 本项目基于PaddleVideo视频开发套件训练打架识别模型，然后将训练好的模型集成到PaddleDetection的PP-Human中，助力行人行为分析。                     | [基于PP-Human的打架识别](https://aistudio.baidu.com/aistudio/projectdetail/4086987?contributionType=1) | [下载链接](https://aistudio.baidu.com/aistudio/projectdetail/4086987?contributionType=1) |
| 智能安防 | 摔倒检测 | 基于PP-Human完成来客分析整体流程。使用PP-Human完成来客分析中非常常见的场景： 1. 来客属性识别(单镜和跨境可视化）；2. 来客行为识别（摔倒识别）。 | [基于PP-Human的来客分析案例教程](https://aistudio.baidu.com/aistudio/projectdetail/4537344)            | [下载链接](https://aistudio.baidu.com/aistudio/projectdetail/4537344)                    |
</details>

### 🏎️PP-Vehicle 实时车辆分析工具

<details>
<summary><b> 简介(点击展开)</b></summary>

PaddleDetection深入探索核心行业的高频场景，提供了车辆开箱即用分析工具，支持图片/单镜头视频/多镜头视频/在线视频流多种输入方式，广泛应用于智慧交通、智慧城市、工业巡检等领域。支持服务器端部署及TensorRT加速，T4服务器上可达到实时。
PP-Vehicle囊括四大交通场景核心功能：车牌识别、属性识别、车流量统计、违章检测。

`传送门`：[PP-Vehicle车辆分析工具指南](deploy/pipeline/README.md)。

</details>

<details>
<summary><b> 预训练模型(点击展开)</b></summary>

|        任务        | T4 TensorRT FP16: 速度(FPS) | 推荐部署硬件 |                                                                                           模型方案                                                                                           |                模型体积                 |
| :----------------: | :-------------------------: | :----------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------: |
| 车辆检测（高精度） |            38.9             |    服务器    |                                                [目标检测](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_ppvehicle.zip)                                                |                  182M                   |
| 车辆跟踪（高精度） |             25              |    服务器    |                                               [多目标跟踪](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_ppvehicle.zip)                                               |                  182M                   |
|      车牌识别      |            213.7            |    服务器    | [车牌检测](https://bj.bcebos.com/v1/paddledet/models/pipeline/ch_PP-OCRv3_det_infer.tar.gz) <br> [车牌识别](https://bj.bcebos.com/v1/paddledet/models/pipeline/ch_PP-OCRv3_rec_infer.tar.gz) | 车牌检测：3.9M  <br> 车牌字符识别： 12M |
|      车辆属性      |            136.8            |    服务器    |                                                  [属性识别](https://bj.bcebos.com/v1/paddledet/models/pipeline/vehicle_attribute_model.zip)                                                  |                  7.2M                   |

`传送门`：[完整预训练模型](deploy/pipeline/README.md)。
</details>

<details>
<summary><b> 产业应用代码示例(点击展开)</b></summary>

| 行业     | 类别             | 亮点                                                                                                               | 文档说明                                                                                      | 模型下载                                                              |
| -------- | ---------------- | ------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| 智慧交通 | 交通监控车辆分析 | 本项目基于PP-Vehicle演示智慧交通中最刚需的车流量监控、车辆违停检测以及车辆结构化（车牌、车型、颜色）分析三大场景。 | [基于PP-Vehicle的交通监控分析系统](https://aistudio.baidu.com/aistudio/projectdetail/4512254) | [下载链接](https://aistudio.baidu.com/aistudio/projectdetail/4512254) |
</details>

## 💡产业实践范例

产业实践范例是PaddleDetection针对高频目标检测应用场景，提供的端到端开发示例，帮助开发者打通数据标注-模型训练-模型调优-预测部署全流程。
针对每个范例我们都通过[AI-Studio](https://ai.baidu.com/ai-doc/AISTUDIO/Tk39ty6ho)提供了项目代码以及说明，用户可以同步运行体验。

`传送门`：[产业实践范例完整列表](industrial_tutorial/README.md)

- [基于PP-YOLOE-R的旋转框检测](https://aistudio.baidu.com/aistudio/projectdetail/5058293)
- [基于PP-YOLOE-SOD的无人机航拍图像检测](https://aistudio.baidu.com/aistudio/projectdetail/5036782)
- [基于PP-Vehicle的交通监控分析系统](https://aistudio.baidu.com/aistudio/projectdetail/4512254)
- [基于PP-Human v2的摔倒检测](https://aistudio.baidu.com/aistudio/projectdetail/4606001)
- [基于PP-TinyPose增强版的智能健身动作识别](https://aistudio.baidu.com/aistudio/projectdetail/4385813)
- [基于PP-Human的打架识别](https://aistudio.baidu.com/aistudio/projectdetail/4086987?contributionType=1)
- [基于Faster-RCNN的瓷砖表面瑕疵检测](https://aistudio.baidu.com/aistudio/projectdetail/2571419)
- [基于PaddleDetection的PCB瑕疵检测](https://aistudio.baidu.com/aistudio/projectdetail/2367089)
- [基于FairMOT实现人流量统计](https://aistudio.baidu.com/aistudio/projectdetail/2421822)
- [基于YOLOv3实现跌倒检测](https://aistudio.baidu.com/aistudio/projectdetail/2500639)
- [基于PP-PicoDetv2 的路面垃圾检测](https://aistudio.baidu.com/aistudio/projectdetail/3846170?channelType=0&channel=0)
- [基于人体关键点检测的合规检测](https://aistudio.baidu.com/aistudio/projectdetail/4061642?contributionType=1)
- [基于PP-Human的来客分析案例教程](https://aistudio.baidu.com/aistudio/projectdetail/4537344)
- 持续更新中...

## 🏆企业应用案例

企业应用案例是企业在实生产环境下落地应用PaddleDetection的方案思路，相比产业实践范例其更多强调整体方案设计思路，可供开发者在项目方案设计中做参考。

`传送门`：[企业应用案例完整列表](https://www.paddlepaddle.org.cn/customercase)

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
- 持续更新中...

## 📝许可证书

本项目的发布受[Apache 2.0 license](LICENSE)许可认证。


## 📌引用

```
@misc{ppdet2019,
title={PaddleDetection, Object detection and instance segmentation toolkit based on PaddlePaddle.},
author={PaddlePaddle Authors},
howpublished = {\url{https://github.com/PaddlePaddle/PaddleDetection}},
year={2019}
}
```
