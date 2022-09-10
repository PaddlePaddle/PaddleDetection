[简体中文](README.md) | English

# Real Time Pedestrian Analysis Tool PP-Human

**PP-Human is the industry's first open-sourced real-time pedestrian analysis tool based on PaddlePaddle deep learning framework. It has three major features: rich functions, wide application, and efficient deployment.**



![](https://user-images.githubusercontent.com/22989727/178965250-14be25c1-125d-4d90-8642-7a9b01fecbe2.gif)



PP-Human supports various inputs such as images, single-camera, and multi-camera videos. It covers multi-object tracking, attributes recognition, behavior analysis, visitor traffic statistics, and trace records. PP-Human can be applied to fields including Smart Transportation, Smart Community, and industrial inspections. It can also be deployed on server sides and TensorRT accelerator. On the T4 server, it could achieve real-time analysis.

## 📣 Updates

- 🔥 **2022.7.13：PP-Human v2 launched with a full upgrade of four industrial features: behavior analysis, attributes recognition, visitor traffic statistics and ReID. It provides a strong core algorithm for pedestrian detection, tracking and attribute analysis with a simple and detailed release/2.5ment process and model optimization strategy.**
- 2022.4.18: Add  PP-Human practical tutorials, including training, deployment, and action expansion. Details for AIStudio project please see [Link](https://aistudio.baidu.com/aistudio/projectdetail/3842982)

- 2022.4.10: Add PP-Human examples; empower refined management of intelligent community management. A quick start for AIStudio [Link](https://aistudio.baidu.com/aistudio/projectdetail/3679564)
- 2022.4.5: Launch the real-time pedestrian analysis tool PP-Human. It supports pedestrian tracking, visitor traffic statistics, attributes recognition, and falling detection. Due to its specific optimization of real-scene data, it can accurately recognize various falling gestures, and adapt to different environmental backgrounds, light and camera angles.

## 🔮 Features and demonstration

| ⭐ Feature                                          | 💟 Advantages                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | 💡Example                                                                                                                                     |
| -------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **ReID**                                           | Extraordinary performance: special optimization for technical challenges such as target occlusion, uncompleted and blurry objects to achieve mAP 98.8, 1.5ms/person                                                                                                                                                                                                                                                                                                                                                    | <img src="https://user-images.githubusercontent.com/48054808/173037607-0a5deadc-076e-4dcc-bd96-d54eea205f1f.png" title="" alt="" width="191"> |
| **Attribute analysis**                             | Compatible with a variety of data formats: support for images, video input<br/><br/>High performance: Integrated open-sourced datasets with real enterprise data for training, achieved mAP 94.86, 2ms/person<br/><br/>Support 26 attributes: gender, age, glasses, tops, shoes, hats, backpacks and other 26 high-frequency attributes                                                                                                                                                                                | <img src="https://user-images.githubusercontent.com/48054808/173036043-68b90df7-e95e-4ada-96ae-20f52bc98d7c.png" title="" alt="" width="207"> |
| **Behaviour detection**                            | Rich function: support five high-frequency anomaly behavior detection of falling, fighting, smoking, telephoning, and intrusion<br/><br/>Robust: unlimited by different environmental backgrounds, light, and camera angles.<br/><br/>High performance: Compared with video recognition technology, it takes significantly smaller computation resources; support localization and service-oriented rapid deployment<br/><br/>Fast training: only takes 15 minutes to produce high precision behavior detection models | <img src="https://user-images.githubusercontent.com/48054808/173034825-623e4f78-22a5-4f14-9b83-dc47aa868478.gif" title="" alt="" width="209"> |
| **Visitor traffic statistics**<br>**Trace record** | Simple and easy to use: single parameter to initiate functions of visitor traffic statistics and trace record                                                                                                                                                                                                                                                                                                                                                                                                          | <img src="https://user-images.githubusercontent.com/22989727/174736440-87cd5169-c939-48f8-90a1-0495a1fcb2b1.gif" title="" alt="" width="200"> |

## 🗳 Model Zoo

<details>
<summary><b>PP-Human End-to-end model results (click to expand)</b></summary>

| Task                                   | End-to-End Speed（ms） | Model                                                                                                                                                                                                                                                                                                                           | Size                                                                                                   |
|:--------------------------------------:|:--------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------:|
| Pedestrian detection (high precision)  | 25.1ms               | [Multi-object tracking](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)                                                                                                                                                                                                                      | 182M                                                                                                   |
| Pedestrian detection (lightweight)     | 16.2ms               | [Multi-object tracking](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_s_36e_pipeline.zip)                                                                                                                                                                                                                      | 27M                                                                                                    |
| Pedestrian tracking (high precision)   | 31.8ms               | [Multi-object tracking](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)                                                                                                                                                                                                                      | 182M                                                                                                   |
| Pedestrian tracking (lightweight)      | 21.0ms               | [Multi-object tracking](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_s_36e_pipeline.zip)                                                                                                                                                                                                                      | 27M                                                                                                    |
| Attribute recognition (high precision) | Single person8.5ms   | [Object detection](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)<br> [Attribute recognition](https://bj.bcebos.com/v1/paddledet/models/pipeline/strongbaseline_r50_30e_pa100k.zip)                                                                                                         | Object detection：182M<br>Attribute recognition：86M                                                     |
| Attribute recognition (lightweight)    | Single person 7.1ms  | [Object detection](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)<br> [Attribute recognition](https://bj.bcebos.com/v1/paddledet/models/pipeline/strongbaseline_r50_30e_pa100k.zip)                                                                                                         | Object detection：182M<br>Attribute recognition：86M                                                     |
| Falling detection                      | Single person 10ms   | [Multi-object tracking](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip) <br> [Keypoint detection](https://bj.bcebos.com/v1/paddledet/models/pipeline/dark_hrnet_w32_256x192.zip) <br> [Behavior detection based on key points](https://bj.bcebos.com/v1/paddledet/models/pipeline/STGCN.zip) | Multi-object tracking：182M<br>Keypoint detection：101M<br>Behavior detection based on key points: 21.8M |
| Intrusion detection                    | 31.8ms               | [Multi-object tracking](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)                                                                                                                                                                                                                      | 182M                                                                                                   |
| Fighting detection                     | 19.7ms               | [Video classification](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)                                                                                                                                                                                                                       | 90M                                                                                                    |
| Smoking detection                      | Single person 15.1ms | [Object detection](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)<br>[Object detection based on Human Id](https://bj.bcebos.com/v1/paddledet/models/pipeline/ppyoloe_crn_s_80e_smoking_visdrone.zip)                                                                                        | Object detection：182M<br>Object detection based on Human ID: 27M                                       |
| Phoning detection                      | Single person ms     | [Object detection](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)<br>[Image classification based on Human ID](https://bj.bcebos.com/v1/paddledet/models/pipeline/PPHGNet_tiny_calling_halfbody.zip)                                                                                         | Object detection：182M<br>Image classification based on Human ID：45M                                    |

</details>

<details>
<summary><b>PP-Vehicle End-to-end model results (click to expand)</b></summary>

| Task                                   | End-to-End Speed（ms） | Model                                                                                                                                                                                                                                                                                                                           | Size                                                                                                   |
|:--------------------------------------:|:--------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------:|
| Vehicle detection (high precision)  | 25.7ms               | [object detection](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_ppvehicle.zip)                                                                                                                                                                                                                      | 182M                                                                                                   |
| Vehicle detection (lightweight)     | 13.2ms               | [object detection](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_s_36e_ppvehicle.zip)                                                                                                                                                                                                                      | 27M                                                                                                    |
| Vehicle tracking (high precision)   | 40ms               | [multi-object tracking](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_ppvehicle.zip)                                                                                                                                                                                                                      | 182M                                                                                                   |
| Vehicle tracking (lightweight)      | 25ms               | [multi-object tracking](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_s_36e_pipeline.zip)                                                                                                                                                                                                                      | 27M                                                                                                    |
| Plate Recognition                   | 4.68ms     | [plate detection](https://bj.bcebos.com/v1/paddledet/models/pipeline/ch_PP-OCRv3_det_infer.tar.gz)<br>[plate recognition](https://bj.bcebos.com/v1/paddledet/models/pipeline/ch_PP-OCRv3_rec_infer.tar.gz)                                                                                         | Plate detection：3.9M<br>Plate recognition：12M                                    |
| Vehicle attribute      | 7.31ms               | [attribute recognition](https://bj.bcebos.com/v1/paddledet/models/pipeline/vehicle_attribute_model.zip)                                                                                                                                                                                                                      | 7.2M                                                                                                    |

</details>


Click to download the model, then unzip and save it in the `. /output_inference`.

## 📚 Doc Tutorials

### [A Quick Start](docs/tutorials/PPHuman_QUICK_STARTED_en.md)

### Pedestrian attribute/feature recognition

* [A quick start](docs/tutorials/pphuman_attribute_en.md)
* [Customized development tutorials](../../docs/advanced_tutorials/customization/pphuman_attribute_en.md)
  * Data Preparation
  * Model Optimization
  * New Attributes

### Behavior detection

* [A quick start](docs/tutorials/pphuman_action_en.md)
  * Falling detection
  * Fighting detection
* [Customized development tutorials](../../docs/advanced_tutorials/customization/action_recognotion/README_en.md)
  * Solution Selection
  * Data Preparation
  * Model Optimization
  * New Attributes

### ReID

* [A quick start](docs/tutorials/pphuman_mtmct_en.md)
* [Customized development tutorials](../../docs/advanced_tutorials/customization/pphuman_mtmct_en.md)
  * Data Preparation
  * Model Optimization

### Pedestrian tracking, visitor traffic statistics, trace records

* [A quick start](docs/tutorials/pphuman_mot_en.md)
  * Pedestrian tracking,
  * Visitor traffic statistics
  * Regional intrusion diagnosis and counting
* [Customized development tutorials](../../docs/advanced_tutorials/customization/pphuman_mot_en.md)
  * Data Preparation
  * Model Optimization
