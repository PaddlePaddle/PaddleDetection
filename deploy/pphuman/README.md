[English](README_en.md) | 简体中文

# 实时行人分析工具 PP-Human

**PP-Human是基于飞桨深度学习框架的业界首个开源产业级实时行人分析工具，具有功能丰富，应用广泛和部署高效三大优势。**

![](https://user-images.githubusercontent.com/48054808/173030254-ecf282bd-2cfe-43d5-b598-8fed29e22020.gif)

PP-Human支持图片/单镜头视频/多镜头视频多种输入方式，功能覆盖多目标跟踪、属性识别、行为分析及人流量计数与轨迹记录。能够广泛应用于智慧交通、智慧社区、工业巡检等领域。支持服务器端部署及TensorRT加速，T4服务器上可达到实时。

## 📣 近期更新

- 2022.4.18：新增PP-Human全流程实战教程, 覆盖训练、部署、动作类型扩展等内容，AIStudio项目请见[链接](https://aistudio.baidu.com/aistudio/projectdetail/3842982)
- 2022.4.10：新增PP-Human范例，赋能社区智能精细化管理, AIStudio快速上手教程[链接](https://aistudio.baidu.com/aistudio/projectdetail/3679564)
- 2022.4.5：全新发布实时行人分析工具PP-Human，支持行人跟踪、人流量统计、人体属性识别与摔倒检测四大能力，基于真实场景数据特殊优化，精准识别各类摔倒姿势，适应不同环境背景、光线及摄像角度

## 🔮 功能介绍与效果展示

| ⭐ 功能           | 💟 方案优势                                                                                                                                           | 💡示例图                                                                                                                                         |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **跨镜跟踪（ReID）** | 超强性能：针对目标遮挡、完整度、模糊度等难点特殊优化，实现mAP 98.8、1.5ms/人                                                                                                     | <img src="https://user-images.githubusercontent.com/48054808/173037607-0a5deadc-076e-4dcc-bd96-d54eea205f1f.png" title="" alt="" width="191"> |
| **属性分析**       | 兼容多种数据格式：支持图片、视频输入<br/><br/>高性能：融合开源数据集与企业真实数据进行训练，实现mAP 94.86、2ms/人<br/><br/>支持26种属性：性别、年龄、眼镜、上衣、鞋子、帽子、背包等26种高频属性                                | <img src="https://user-images.githubusercontent.com/48054808/173036043-68b90df7-e95e-4ada-96ae-20f52bc98d7c.png" title="" alt="" width="207"> |
| **行为识别**       | 功能丰富：支持摔倒、打架、抽烟、打电话、人员闯入五种高频异常行为识别<br/><br/>鲁棒性强：对光照、视角、背景环境无限制<br/><br/>性能高：与视频识别技术相比，模型计算量大幅降低，支持本地化与服务化快速部署<br/><br/>训练速度快：仅需15分钟即可产出高精度行为识别模型 | <img src="https://user-images.githubusercontent.com/48054808/173034825-623e4f78-22a5-4f14-9b83-dc47aa868478.gif" title="" alt="" width="209"> |
| **人流量计数与轨迹记录** | 简洁易用：单个参数即可开启人流量计数与轨迹记录功能                                                                                                                         | <img src="https://user-images.githubusercontent.com/22989727/174736440-87cd5169-c939-48f8-90a1-0495a1fcb2b1.gif" title="" alt="" width="200"> |

## 🗳 模型库

| 任务    | 适用场景         | 精度         | 预测速度（ms） | 模型权重                                                                                           | 预测部署模型                                                                                       |
|:-----:|:------------:|:---------- |:--------:|:----------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------:|
| 目标检测  | 图片输入         | mAP: 56.3  | 28.0ms   | [下载链接](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.pdparams) | [下载链接](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)    |
| 目标跟踪  | 视频输入         | MOTA: 72.0 | 33.1ms   | [下载链接](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.pdparams) | [下载链接](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)    |
| 属性识别  | 图片/视频输入 属性识别 | mA: 94.86  | 单人2ms    | -                                                                                              | [下载链接](https://bj.bcebos.com/v1/paddledet/models/pipeline/strongbaseline_r50_30e_pa100k.zip) |
| 关键点检测 | 视频输入 行为识别    | AP: 87.1   | 单人2.9ms  | [下载链接](https://bj.bcebos.com/v1/paddledet/models/pipeline/dark_hrnet_w32_256x192.pdparams)     | [下载链接](https://bj.bcebos.com/v1/paddledet/models/pipeline/dark_hrnet_w32_256x192.zip)        |
| 行为识别  | 视频输入 行为识别    | 准确率: 96.43 | 单人2.7ms  | -                                                                                              | [下载链接](https://bj.bcebos.com/v1/paddledet/models/pipeline/STGCN.zip)                         |
| ReID  | 视频输入 跨镜跟踪    | mAP: 98.8  | 单人1.5ms  | -                                                                                              | [下载链接](https://bj.bcebos.com/v1/paddledet/models/pipeline/reid_model.zip)                    |

## 📚 文档教程

### [快速开始](docs/tutorials/QUICK_STARTED.md)

### 行人属性/特征识别

* [快速开始](docs/tutorials/attribute.md)
* [二次开发教程](../../docs/advanced_tutorials/customization/attribute.md)
  * 数据准备
  * 模型优化
  * 新增属性

### 行为识别

* [快速开始](docs/tutorials/action.md)
  * 摔倒检测
  * 打架识别
* [二次开发教程](../../docs/advanced_tutorials/customization/action.md)
  * 方案选择
  * 数据准备
  * 模型优化
  * 新增行为

### 跨镜跟踪ReID

* [快速开始](docs/tutorials/mtmct.md)
* [二次开发教程]()
  * 数据准备
  * 模型优化

### 人流量计数与轨迹记录

* [快速开始](docs/tutorials/mot.md)
* [二次开发教程](../../docs/advanced_tutorials/customization/mot.md)
  * 数据准备
  * 模型优化
