English | [简体中文](README.md)

# Real-Time Pedestrian Analysis Tool ———— PP-Human

PP-Human serves as the first open-source tool of real-time pedestrian anaylsis relying on the PaddlePaddle deep learning framework. It has three advantages: rich functions, wide application and efficient deployment.

![](https://user-images.githubusercontent.com/48054808/173030254-ecf282bd-2cfe-43d5-b598-8fed29e22020.gif)

PP-Human offers many input options, including image/single-camera video/multi-camera video, and covers multi-object tracking, attribute recognition, and action recognition. PP-Human can be applied to intelligent traffic, the intelligent community, industiral patrol, and so on. It supports server-side deployment and TensorRT acceleration，and achieves real-time analysis on the T4 server.

## 📣 Recent updates

- 2022.4.18：Full-process operation tutorial of PP-Human, covering training, deployment, action expansion, please refer to this [AI Studio project](https://aistudio.baidu.com/aistudio/projectdetail/3842982).

- 2022.4.10：Community intelligent management supportted by PP-Human, please refer to this [AI Studio project](https://aistudio.baidu.com/aistudio/projectdetail/3679564) for quick start tutorial.
- 2022.4.5：The real-time pedestrian analysis tool PP-Human is released, which supports four capabilities: pedestrian tracking, pedestrian flow statistics, human attribute recognition and fall detection. It is specially optimized based on real scene data to accurately identify various fall postures and adapt to different environmental backgrounds, light and camera angles.


## 🔮 Function introduction and effect display

| ⭐ 功能           | 💟 方案优势                                                                                                                                           | 💡示例图                                                                                                                                         |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **跨镜跟踪（ReID）** | 超强性能：针对目标遮挡、完整度、模糊度等难点特殊优化，实现mAP 98.8、1.5ms/人                                                                                                     | <img src="https://user-images.githubusercontent.com/48054808/173037607-0a5deadc-076e-4dcc-bd96-d54eea205f1f.png" title="" alt="" width="191"> |
| **属性分析**       | 兼容多种数据格式：支持图片、视频输入<br/><br/>高性能：融合开源数据集与企业真实数据进行训练，实现mAP 94.86、2ms/人<br/><br/>支持26种属性：性别、年龄、眼镜、上衣、鞋子、帽子、背包等26种高频属性                                | <img src="https://user-images.githubusercontent.com/48054808/173036043-68b90df7-e95e-4ada-96ae-20f52bc98d7c.png" title="" alt="" width="207"> |
| **行为识别**       | 功能丰富：支持摔倒、打架、抽烟、打电话、人员闯入五种高频异常行为识别<br/><br/>鲁棒性强：对光照、视角、背景环境无限制<br/><br/>性能高：与视频识别技术相比，模型计算量大幅降低，支持本地化与服务化快速部署<br/><br/>训练速度快：仅需15分钟即可产出高精度行为识别模型 | <img src="https://user-images.githubusercontent.com/48054808/173034825-623e4f78-22a5-4f14-9b83-dc47aa868478.gif" title="" alt="" width="209"> |
| **人流量计数与轨迹记录** | 简洁易用：单个参数即可开启人流量计数与轨迹记录功能                                                                                                                         | <img src="https://user-images.githubusercontent.com/22989727/174736440-87cd5169-c939-48f8-90a1-0495a1fcb2b1.gif" title="" alt="" width="200"> |


## 🗳 Model ZOO

| Task            | Scenario | Precision | Inference Speed（FPS） | Model Weights |Model Inference and Deployment |
| :---------:     |:---------:     |:---------------     | :-------:  | :------:      | :------:      |
| Object Detection(high-precision)        | Image/Video Input | mAP: 56.6  | 28.0ms           |[Link](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.pdparams) |[Link](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip) |
| Object Detection(light-weight)        | Image/Video Input | mAP: 53.2  | 22.1ms           |[Link](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_s_36e_pipeline.pdparams) |[Link](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_s_36e_pipeline.zip) |
| Object Tracking(high-precision)       | Image/Video Input | MOTA: 79.5  | 33.1ms           |[Link](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.pdparams) |[Link](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip) |
| Object Tracking(light-weight)       | Image/Video Input | MOTA: 69.1  | 27.2ms           |[Link](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_s_36e_pipeline.pdparams) |[Link](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_s_36e_pipeline.zip) |
| Attribute Recognition    | Image/Video Input  Attribute Recognition | mA: 94.86 |  2ms per person       | - |[Link](https://bj.bcebos.com/v1/paddledet/models/pipeline/strongbaseline_r50_30e_pa100k.zip) |
| Keypoint Detection    | Video Input  Falling Recognition | AP: 87.1 | 2.9ms per person        | [Link](https://bj.bcebos.com/v1/paddledet/models/pipeline/dark_hrnet_w32_256x192.pdparams) |[Link](https://bj.bcebos.com/v1/paddledet/models/pipeline/dark_hrnet_w32_256x192.zip)
| Falling Recognition   |  Video Input  Falling Recognition  | Precision 96.43 |  2.7ms per person          | - |[Link](https://bj.bcebos.com/v1/paddledet/models/pipeline/STGCN.zip) |
| ReID         | Multi-Target Multi-Camera Tracking   | mAP: 98.8 | 1.5ms per person    | - |[Link](https://bj.bcebos.com/v1/paddledet/models/pipeline/reid_model.zip) |

Then, unzip the downloaded model to the folder `./output_inference`.


## 📚 Documentation tutorial
### [QUICK START](docs/tutorials/QUICK_STARTED.md)


### Pedestrian Attribute/Feature Recognition

* [QUICK START](docs/tutorials/attribute.md)

* [Development Tutorial](../../docs/advanced_tutorials/customization/attribute.md)
  * Data preparation
  * Model optimization
  * Add new attribute

### Action Recognition

* [QUICK START](docs/tutorials/action.md)
  * Fall detection

* [Development Tutorial](../../docs/advanced_tutorials/customization/action.md)
  * Scheme selection
  * Data preparation
  * Model optimization
  * Add new action


### Multi-Target Multi-Camera Tracking and ReID

* [QUICK START](docs/tutorials/mtmct.md)

* [Development Tutorial]()
  * Data preparation
  * Model optimization


### Passenger flow counting and track recording

* [QUICK START](docs/tutorials/mot.md)

* [Development Tutorial](../../docs/advanced_tutorials/customization/mot.md)
  * Data preparation
  * Model optimization
