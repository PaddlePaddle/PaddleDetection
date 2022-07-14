简体中文

# 实时行人分析工具 PP-Human

**PP-Human是基于飞桨深度学习框架的业界首个开源产业级实时行人分析工具，具有功能丰富，应用广泛和部署高效三大优势。**

![](https://user-images.githubusercontent.com/22989727/178892756-e2717a2c-beb0-4d88-ad32-ca37e24b47f8.gif)

PP-Human支持图片/单镜头视频/多镜头视频多种输入方式，功能覆盖多目标跟踪、属性识别、行为分析及人流量计数与轨迹记录。能够广泛应用于智慧交通、智慧社区、工业巡检等领域。支持服务器端部署及TensorRT加速，T4服务器上可达到实时。

## 📣 近期更新

- 🔥 **2022.7.13：PP-Human v2发布，行为识别、人体属性识别、流量计数、跨镜跟踪四大产业特色功能全面升级，覆盖行人检测、跟踪、属性三类核心算法能力，提供保姆级全流程开发及模型优化策略。**
- 2022.4.18：新增PP-Human全流程实战教程, 覆盖训练、部署、动作类型扩展等内容，AIStudio项目请见[链接](https://aistudio.baidu.com/aistudio/projectdetail/3842982)
- 2022.4.10：新增PP-Human范例，赋能社区智能精细化管理, AIStudio快速上手教程[链接](https://aistudio.baidu.com/aistudio/projectdetail/3679564)
- 2022.4.5：全新发布实时行人分析工具PP-Human，支持行人跟踪、人流量统计、人体属性识别与摔倒检测四大能力，基于真实场景数据特殊优化，精准识别各类摔倒姿势，适应不同环境背景、光线及摄像角度

## 🔮 功能介绍与效果展示

| ⭐ 功能           | 💟 方案优势                                                                                                                                           | 💡示例图                                                                                                                                         |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **跨镜跟踪（ReID）** | 超强性能：针对目标遮挡、完整度、模糊度等难点特殊优化，实现mAP 98.8、1.5ms/人                                                                                                     | <img src="https://user-images.githubusercontent.com/48054808/173037607-0a5deadc-076e-4dcc-bd96-d54eea205f1f.png" title="" alt="" width="191"> |
| **属性分析**       | 兼容多种数据格式：支持图片、视频输入<br/><br/>高性能：融合开源数据集与企业真实数据进行训练，实现mAP 94.86、2ms/人<br/><br/>支持26种属性：性别、年龄、眼镜、上衣、鞋子、帽子、背包等26种高频属性                                | <img src="https://user-images.githubusercontent.com/48054808/173036043-68b90df7-e95e-4ada-96ae-20f52bc98d7c.png" title="" alt="" width="207"> |
| **行为识别**       | 功能丰富：支持摔倒、打架、抽烟、打电话、人员闯入五种高频异常行为识别<br/><br/>鲁棒性强：对光照、视角、背景环境无限制<br/><br/>性能高：与视频识别技术相比，模型计算量大幅降低，支持本地化与服务化快速部署<br/><br/>训练速度快：仅需15分钟即可产出高精度行为识别模型 | <img src="https://user-images.githubusercontent.com/48054808/173034825-623e4f78-22a5-4f14-9b83-dc47aa868478.gif" title="" alt="" width="209"> |
| **人流量计数**<br>**轨迹记录** | 简洁易用：单个参数即可开启人流量计数与轨迹记录功能                                                                                                                         | <img src="https://user-images.githubusercontent.com/22989727/174736440-87cd5169-c939-48f8-90a1-0495a1fcb2b1.gif" title="" alt="" width="200"> |

## 🗳 模型库

<details>
<summary><b>单模型效果（点击展开）</b></summary>

| 任务            | 适用场景 | 精度 | 预测速度（ms）| 模型体积 | 预测部署模型 |
| :---------:     |:---------:     |:---------------     | :-------:  |  :------:      | :------:      |
| 目标检测(高精度) | 图片输入 | mAP: 57.8  | 25.1ms          | 182M |[下载链接](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip) |
| 目标检测(轻量级) | 图片输入 | mAP: 53.2  | 16.2ms          | 27M |[下载链接](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_s_36e_pipeline.zip) |
| 目标跟踪(高精度) | 视频输入 | MOTA: 82.2  | 31.8ms           | 182M |[下载链接](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip) |
| 目标跟踪(轻量级) | 视频输入 | MOTA: 73.9  | 21.0ms           |27M |[下载链接](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_s_36e_pipeline.zip) |
| 属性识别（高精度）    | 图片/视频输入 属性识别  | mA: 95.4 |  单人4.2ms     | 86M |[下载链接](https://bj.bcebos.com/v1/paddledet/models/pipeline/PPHGNet_small_person_attribute_954_infer.zip) |
| 属性识别（轻量级）    | 图片/视频输入 属性识别  | mA: 94.5 |  单人2.9ms     | 7.2M |[下载链接](https://bj.bcebos.com/v1/paddledet/models/pipeline/PPLCNet_x1_0_person_attribute_945_infer.zip) |
| 关键点检测    | 视频输入 行为识别 | AP: 87.1 | 单人5.7ms        | 101M |[下载链接](https://bj.bcebos.com/v1/paddledet/models/pipeline/dark_hrnet_w32_256x192.zip) |
| 基于关键点序列分类   |  视频输入 行为识别  | 准确率: 96.43 |  单人0.07ms      | 21.8M |[下载链接](https://bj.bcebos.com/v1/paddledet/models/pipeline/STGCN.zip) |
| 基于人体id图像分类 |  视频输入 行为识别  | 准确率: 86.85 |  单人1.8ms      | 45M |[下载链接](https://bj.bcebos.com/v1/paddledet/models/pipeline/PPHGNet_tiny_calling_halfbody.zip) |
| 基于人体id检测 |  视频输入 行为识别  | AP50: 79.5 |  单人10.9ms      | 27M |[下载链接](https://bj.bcebos.com/v1/paddledet/models/pipeline/ppyoloe_crn_s_80e_smoking_visdrone.zip) |
|    视频分类    |  视频输入 行为识别  | Accuracy： 89.0 | 19.7ms/1s视频 | 90M | [下载链接](https://videotag.bj.bcebos.com/PaddleVideo-release2.3/ppTSM_fight.pdparams) |
| ReID         | 视频输入 跨镜跟踪   | mAP: 98.8 | 单人0.23ms   | 85M |[下载链接](https://bj.bcebos.com/v1/paddledet/models/pipeline/reid_model.zip) |

</details>

<details>
<summary><b>端到端模型效果（点击展开）</b></summary>

| 任务            | 端到端速度（ms）|  模型方案  |  模型体积 |
| :---------:     | :-------:  |  :------: |:------: |
|  行人检测（高精度）  | 25.1ms  |  [多目标跟踪](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip) | 182M |  
|  行人检测（轻量级）  | 16.2ms  |  [多目标跟踪](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_s_36e_pipeline.zip) | 27M |
|  行人跟踪（高精度）  | 31.8ms  |  [多目标跟踪](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip) | 182M |  
|  行人跟踪（轻量级）  | 21.0ms  |  [多目标跟踪](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_s_36e_pipeline.zip) | 27M |
|  属性识别（高精度）  |   单人8.5ms | [目标检测](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)<br> [属性识别](https://bj.bcebos.com/v1/paddledet/models/pipeline/strongbaseline_r50_30e_pa100k.zip) | 目标检测：182M<br>属性识别：86M |
|  属性识别（轻量级）  |   单人7.1ms | [目标检测](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)<br> [属性识别](https://bj.bcebos.com/v1/paddledet/models/pipeline/strongbaseline_r50_30e_pa100k.zip) | 目标检测：182M<br>属性识别：86M |
|  摔倒识别  |   单人10ms | [多目标跟踪](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip) <br> [关键点检测](https://bj.bcebos.com/v1/paddledet/models/pipeline/dark_hrnet_w32_256x192.zip) <br> [基于关键点行为识别](https://bj.bcebos.com/v1/paddledet/models/pipeline/STGCN.zip) | 多目标跟踪：182M<br>关键点检测：101M<br>基于关键点行为识别：21.8M |
|  闯入识别  |   31.8ms | [多目标跟踪](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip) | 182M |
|  打架识别  |   19.7ms | [视频分类](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip) | 90M |
|  抽烟识别  |   单人15.1ms | [目标检测](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)<br>[基于人体id的目标检测](https://bj.bcebos.com/v1/paddledet/models/pipeline/ppyoloe_crn_s_80e_smoking_visdrone.zip) | 目标检测：182M<br>基于人体id的目标检测：27M |
|  打电话识别  |   单人ms | [目标检测](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)<br>[基于人体id的图像分类](https://bj.bcebos.com/v1/paddledet/models/pipeline/PPHGNet_tiny_calling_halfbody.zip) | 目标检测：182M<br>基于人体id的图像分类：45M |

 </details>


点击模型方案中的模型即可下载指定模型，下载后解压存放至`./output_inference`目录中

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
* [二次开发教程](../../docs/advanced_tutorials/customization/action_recognotion/README.md)
  * 方案选择
  * 数据准备
  * 模型优化
  * 新增行为

### 跨镜跟踪ReID

* [快速开始](docs/tutorials/mtmct.md)
* [二次开发教程](../../docs/advanced_tutorials/customization/mtmct.md)
  * 数据准备
  * 模型优化

### 行人跟踪、人流量计数与轨迹记录

* [快速开始](docs/tutorials/mot.md)
  * 行人跟踪
  * 人流量计数与轨迹记录
  * 区域闯入判断和计数
* [二次开发教程](../../docs/advanced_tutorials/customization/mot.md)
  * 数据准备
  * 模型优化
