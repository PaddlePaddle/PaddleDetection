简体中文 | [English](README_en.md)

<img src="https://user-images.githubusercontent.com/48054808/185032511-0c97b21c-8bab-4ab1-89ee-16e5e81c22cc.png" title="" alt="" data-align="center">



**PaddleDetection深入探索核心行业的高频场景，提供了行人、车辆场景的开箱即用分析工具，支持图片/单镜头视频/多镜头视频/在线视频流多种输入方式，广泛应用于智慧交通、智慧城市、工业巡检等领域。支持服务器端部署及TensorRT加速，T4服务器上可达到实时。**

- 🚶‍♂️🚶‍♀️ **PP-Human支持四大产业级功能：五大异常行为识别、26种人体属性分析、实时人流计数、跨镜头（ReID）跟踪。**

- 🚗🚙 **PP-Vehicle囊括四大交通场景核心功能：车牌识别、属性识别、车流量统计、违章检测。**

![](https://user-images.githubusercontent.com/22989727/202134414-713a00d6-a0a4-4a77-b6e8-05cdb5d42b1e.gif)

## 📣 近期更新

- 🔥🔥🔥 **2023.02.15: Jetson部署专用小模型PP-YOLOE-PLUS-Tiny发布，可在AGX平台实现4路视频流实时预测；PP-Vehicle发布违法分析功能车辆逆行和压车道线**
- **2022.8.20：PP-Vehicle首发，提供车牌识别、车辆属性分析（颜色、车型）、车流量统计以及违章检测四大功能，完善的文档教程支持高效完成二次开发与模型优化**
- **2022.7.13：PP-Human v2发布，新增打架、打电话、抽烟、闯入四大行为识别，底层算法性能升级，覆盖行人检测、跟踪、属性三类核心算法能力，提供保姆级全流程开发及模型优化策略**
- 2022.4.18：新增PP-Human全流程实战教程, 覆盖训练、部署、动作类型扩展等内容，AIStudio项目请见[链接](https://aistudio.baidu.com/aistudio/projectdetail/3842982)
- 2022.4.10：新增PP-Human范例，赋能社区智能精细化管理, AIStudio快速上手教程[链接](https://aistudio.baidu.com/aistudio/projectdetail/3679564)
- 2022.4.5：全新发布实时行人分析工具PP-Human，支持行人跟踪、人流量统计、人体属性识别与摔倒检测四大能力，基于真实场景数据特殊优化，精准识别各类摔倒姿势，适应不同环境背景、光线及摄像角度

## 🔮 功能介绍与效果展示

### PP-Human

| ⭐ 功能                  | 💟 方案优势                                                                                                                                     | 💡示例图                                                                                                      |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| **跨镜跟踪（ReID）**        | 超强性能：针对目标遮挡、完整度、模糊度等难点特殊优化，实现mAP 98.8、1.5ms/人                                                                                               | <img title="" src="https://user-images.githubusercontent.com/48054808/173037607-0a5deadc-076e-4dcc-bd96-d54eea205f1f.png" alt="" width="191"> |
| **属性分析**              | 兼容多种数据格式：支持图片、视频、在线视频流输入<br><br>高性能：融合开源数据集与企业真实数据进行训练，实现mAP 95.4、2ms/人<br><br>支持26种属性：性别、年龄、眼镜、上衣、鞋子、帽子、背包等26种高频属性                         | <img title="" src="https://user-images.githubusercontent.com/48054808/173036043-68b90df7-e95e-4ada-96ae-20f52bc98d7c.png" alt="" width="191">|
| **行为识别（包含摔倒、打架、抽烟、打电话、人员闯入）**              | 功能丰富：支持摔倒、打架、抽烟、打电话、人员闯入五种高频异常行为识别<br><br>鲁棒性强：对光照、视角、背景环境无限制<br><br>性能高：与视频识别技术相比，模型计算量大幅降低，支持本地化与服务化快速部署<br><br>训练速度快：仅需15分钟即可产出高精度行为识别模型 |<img title="" src="https://user-images.githubusercontent.com/48054808/173034825-623e4f78-22a5-4f14-9b83-dc47aa868478.gif" alt="" width="191">  |
| **人流量计数**<br>**轨迹记录** | 简洁易用：单个参数即可开启人流量计数与轨迹记录功能                                                                                                                   | <img title="" src="https://user-images.githubusercontent.com/22989727/174736440-87cd5169-c939-48f8-90a1-0495a1fcb2b1.gif" alt="" width="191"> |

### PP-Vehicle

| ⭐ 功能       | 💟 方案优势                                                                                    | 💡示例图                                                                                                                                         |
| ---------- | ------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **车牌识别**   | 支持传统车牌和新能源绿色车牌 <br/><br/> 车牌识别采用长间隔采样识别与多次结果统计投票方式，算力消耗少，识别精度高，结果稳定性好。 检测模型 hmean: 0.979; 识别模型 acc: 0.773          | <img title="" src="https://user-images.githubusercontent.com/48054808/185027987-6144cafd-0286-4c32-8425-7ab9515d1ec3.png" alt="" width="191"> |
| **车辆属性分析** | 支持多种车型、颜色类别识别 <br/><br/> 使用更强力的Backbone模型PP-HGNet、PP-LCNet,精度高、速度快。识别精度: 90.81 | <img title="" src="https://user-images.githubusercontent.com/48054808/185044490-00edd930-1885-4e79-b3d4-3a39a77dea93.gif" alt="" width="207"> |
| **违章检测**   | 简单易用：一行命令即可实现违停检测，自定义设置区域 <br/><br/> 检测、跟踪效果好，可实现违停车辆车牌识别    | <img title="" src="https://user-images.githubusercontent.com/48054808/185028419-58ae0af8-a035-42e7-9583-25f5e4ce0169.png" alt="" width="209"> |
| **车流量计数**  | 简单易用：一行命令即可开启功能，自定义出入位置 <br/><br/> 可提供目标跟踪轨迹显示，统计准确度高        | <img title="" src="https://user-images.githubusercontent.com/48054808/185028798-9e07379f-7486-4266-9d27-3aec943593e0.gif" alt="" width="200"> |
| **违法分析-车辆逆行**   | 简单易用：一行命令即可开启功能 <br/><br/> 车道线分割使用高精度模型PP-LIteSeg    | <img title="" src="https://raw.githubusercontent.com/LokeZhou/PaddleDetection/develop/deploy/pipeline/docs/images/vehicle_retrograde.gif" alt="" width="200"> |
| **违法分析-压车道线**  | 简单易用：一行命令即可开启功能 <br/><br/> 车道线分割使用高精度模型PP-LIteSeg    | <img title="" src="https://raw.githubusercontent.com/LokeZhou/PaddleDetection/develop/deploy/pipeline/docs/images/vehicle_press.gif" alt="" width="200"> |

## 🗳 模型库

### PP-Human

<details>
<summary><b>端到端模型效果（点击展开）</b></summary>

| 任务        | 端到端速度（ms） | 模型方案                                                                                                                                                                                                                                                                  | 模型体积                                        |
|:---------:|:---------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------:|
| 行人检测（高精度） | 25.1ms    | [多目标跟踪](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)                                                                                                                                                                            | 182M                                        |
| 行人检测（轻量级） | 16.2ms    | [多目标跟踪](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_s_36e_pipeline.zip)                                                                                                                                                                            | 27M                                         |
| 行人检测（超轻量级） | 10ms(Jetson AGX)    | [多目标跟踪](https://bj.bcebos.com/v1/paddledet/models/pipeline/pphuman/ppyoloe_plus_crn_t_auxhead_320_60e_pphuman.tar.gz)                                                                        | 17M                                         |
| 行人跟踪（高精度） | 31.8ms    | [多目标跟踪](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)                                                                                                                                                                            | 182M                                        |
| 行人跟踪（轻量级） | 21.0ms    | [多目标跟踪](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_s_36e_pipeline.zip)                                                                                                                                                                            | 27M                                         |
| 行人跟踪（超轻量级） | 13.2ms(Jetson AGX)    | [多目标跟踪](https://bj.bcebos.com/v1/paddledet/models/pipeline/pphuman/ppyoloe_plus_crn_t_auxhead_320_60e_pphuman.tar.gz)                                                                        | 17M                                         |
|  跨镜跟踪(REID)   |   单人1.5ms | [REID](https://bj.bcebos.com/v1/paddledet/models/pipeline/reid_model.zip) | REID：92M |
| 属性识别（高精度） | 单人8.5ms   | [目标检测](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)<br> [属性识别](https://bj.bcebos.com/v1/paddledet/models/pipeline/strongbaseline_r50_30e_pa100k.zip)                                                                            | 目标检测：182M<br>属性识别：86M                       |
| 属性识别（轻量级） | 单人7.1ms   | [目标检测](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)<br> [属性识别](https://bj.bcebos.com/v1/paddledet/models/pipeline/strongbaseline_r50_30e_pa100k.zip)                                                                            | 目标检测：182M<br>属性识别：86M                       |
| 摔倒识别      | 单人10ms    | [多目标跟踪](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip) <br> [关键点检测](https://bj.bcebos.com/v1/paddledet/models/pipeline/dark_hrnet_w32_256x192.zip) <br> [基于关键点行为识别](https://bj.bcebos.com/v1/paddledet/models/pipeline/STGCN.zip) | 多目标跟踪：182M<br>关键点检测：101M<br>基于关键点行为识别：21.8M |
| 闯入识别      | 31.8ms    | [多目标跟踪](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)                                                                                                                                                                            | 182M                                        |
| 打架识别      | 19.7ms    | [视频分类](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)                                                                                                                                                                             | 90M                                         |
| 抽烟识别      | 单人15.1ms  | [目标检测](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)<br>[基于人体id的目标检测](https://bj.bcebos.com/v1/paddledet/models/pipeline/ppyoloe_crn_s_80e_smoking_visdrone.zip)                                                                 | 目标检测：182M<br>基于人体id的目标检测：27M                |
| 打电话识别     | 单人ms      | [目标检测](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)<br>[基于人体id的图像分类](https://bj.bcebos.com/v1/paddledet/models/pipeline/PPHGNet_tiny_calling_halfbody.zip)                                                                      | 目标检测：182M<br>基于人体id的图像分类：45M                |


点击模型方案中的模型即可下载指定模型，下载后解压存放至`./output_inference`目录中

</details>

### PP-Vehicle

<details>
<summary><b>端到端模型效果（点击展开）</b></summary>

| 任务            | 端到端速度（ms）|  模型方案  |  模型体积 |
| :---------:     | :-------:  |  :------: |:------: |
|  车辆检测（高精度）  | 25.7ms  |  [多目标跟踪](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_ppvehicle.zip) | 182M |  
|  车辆检测（轻量级）  | 13.2ms  |  [多目标跟踪](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_s_36e_ppvehicle.zip) | 27M |
|  车辆检测（超轻量级）  | 10ms（Jetson AGX）  |  [多目标跟踪](https://bj.bcebos.com/v1/paddledet/models/pipeline/ppvehicle/ppyoloe_plus_crn_t_auxhead_320_60e_ppvehicle.tar.gz) | 17M |
|  车辆跟踪（高精度）  | 40ms  |  [多目标跟踪](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_ppvehicle.zip) | 182M |
|  车辆跟踪（轻量级）  | 25ms  |  [多目标跟踪](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_s_36e_ppvehicle.zip) | 27M |
|  车辆跟踪（超轻量级）  | 13.2ms（Jetson AGX）  |  [多目标跟踪](https://bj.bcebos.com/v1/paddledet/models/pipeline/ppvehicle/ppyoloe_plus_crn_t_auxhead_320_60e_ppvehicle.tar.gz) | 17M |
|  车牌识别  |   4.68ms |  [车牌检测](https://bj.bcebos.com/v1/paddledet/models/pipeline/ch_PP-OCRv3_det_infer.tar.gz) <br> [车牌字符识别](https://bj.bcebos.com/v1/paddledet/models/pipeline/ch_PP-OCRv3_rec_infer.tar.gz) | 车牌检测：3.9M  <br> 车牌字符识别： 12M |
|  车辆属性  |   7.31ms | [车辆属性](https://bj.bcebos.com/v1/paddledet/models/pipeline/vehicle_attribute_model.zip) | 7.2M |
|  车道线检测  |   47ms | [车道线模型](https://bj.bcebos.com/v1/paddledet/models/pipeline/pp_lite_stdc2_bdd100k.zip) | 47M |

点击模型方案中的模型即可下载指定模型，下载后解压存放至`./output_inference`目录中

</details>

## 📚 详细文档

### 🚶‍♀️ 行人分析工具PP-Human

#### [快速开始](docs/tutorials/PPHuman_QUICK_STARTED.md)

#### 行为识别

- [快速开始](docs/tutorials/pphuman_action.md)

- [二次开发教程](../../docs/advanced_tutorials/customization/action_recognotion/README.md)

#### 行人属性/特征识别

- [快速开始](docs/tutorials/pphuman_attribute.md)

- [二次开发教程](../../docs/advanced_tutorials/customization/pphuman_attribute.md)

#### 跨镜跟踪/ReID

- [快速开始](docs/tutorials/pphuman_mtmct.md)

- [二次开发教程](../../docs/advanced_tutorials/customization/pphuman_mtmct.md)

#### 行人跟踪、人流计数与轨迹记录

- [快速开始](docs/tutorials/pphuman_mot.md)

- [二次开发教程](../../docs/advanced_tutorials/customization/pphuman_mot.md)

### 🚘 车辆分析工具PP-Vehicle

#### [快速开始](docs/tutorials/PPVehicle_QUICK_STARTED.md)

#### 车牌识别

- [快速开始](docs/tutorials/ppvehicle_plate.md)

- [二次开发教程](../../docs/advanced_tutorials/customization/ppvehicle_plate.md)

#### 车辆属性分析

- [快速开始](docs/tutorials/ppvehicle_attribute.md)

- [二次开发教程](../../docs/advanced_tutorials/customization/ppvehicle_attribute.md)

#### 违章检测

- [快速开始](docs/tutorials/ppvehicle_illegal_parking.md)

- [二次开发教程](../../docs/advanced_tutorials/customization/pphuman_mot.md)

#### 车辆跟踪、车流计数与轨迹记录

- [快速开始](docs/tutorials/ppvehicle_mot.md)

- [二次开发教程](../../docs/advanced_tutorials/customization/pphuman_mot.md)

#### 车辆违法压线

- [快速开始](docs/tutorials/ppvehicle_press.md)

- [二次开发教程](../../docs/advanced_tutorials/customization/ppvehicle_violation.md)

#### 车辆逆行

- [快速开始](docs/tutorials/ppvehicle_retrograde.md)

- [二次开发教程](../../docs/advanced_tutorials/customization/ppvehicle_violation.md)
