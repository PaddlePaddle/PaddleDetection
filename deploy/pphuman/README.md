[English](README_en.md) | 简体中文

# 实时行人分析工具 PP-Human

PP-Human是基于飞桨深度学习框架的业界首个开源的实时行人分析工具，具有功能丰富，应用广泛和部署高效三大优势。PP-Human
支持图片/单镜头视频/多镜头视频多种输入方式，功能覆盖多目标跟踪、属性识别和行为分析。能够广泛应用于智慧交通、智慧社区、工业巡检等领域。支持服务器端部署及TensorRT加速，T4服务器上可达到实时。

## 近期更新

- 新增PP-Human全流程实战教程, 覆盖训练、部署、动作类型扩展等内容，AIStudio项目请见[链接](https://aistudio.baidu.com/aistudio/projectdetail/3842982)
- 新增PP-Human范例，赋能社区智能精细化管理, AIStudio快速上手教程[链接](https://aistudio.baidu.com/aistudio/projectdetail/3679564)
- 全新发布实时行人分析工具PP-Human，支持行人跟踪、人流量统计、人体属性识别与摔倒检测四大能力，基于真实场景数据特殊优化，精准识别各类摔倒姿势，适应不同环境背景、光线及摄像角度。


## 功能介绍与效果展示

## 文档教程

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
  * [二次开发教程](../../docs/advanced_tutorials/customization/action.md)
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
