[简体中文](README.md) | English

<img src="https://user-images.githubusercontent.com/48054808/185032511-0c97b21c-8bab-4ab1-89ee-16e5e81c22cc.png" title="" alt="" data-align="center">

**PaddleDetection has provide out-of-the-box tools in pedestrian and vehicle analysis, and it support multiple input format such as images/videos/multi-videos/online video streams. This make it popular in smart-city\smart transportation and so on. It can be deployed easily with GPU server and TensorRT, which achieves real-time performace.**

- 🚶‍♂️🚶‍♀️ **PP-Human has four major toolbox for pedestrian analysis: five example of behavior analysis、26 attributes recognition、in-out counting、multi-target-multi-camera tracking(REID).**

- 🚗🚙 **PP-Vehicle has four major toolbox for vehicle analysis: The license plate recognition、vechile attributes、in-out counting、illegal_parking recognition.**

![](https://user-images.githubusercontent.com/22989727/202134414-713a00d6-a0a4-4a77-b6e8-05cdb5d42b1e.gif)

## 📣 Updates

- 🔥🔥🔥 **PP-YOLOE-PLUS-Tiny was launched for Jetson deploy, which has achieved 20fps while four rtsp streams work at the same time; PP-Vehicle was launched with retrograde and lane line press.**
- 🔥 **2022.8.20：PP-Vehicle was first launched with four major toolbox for vehicle analysis，and it also provide detailed documentation for user to train with their own datas and model optimize.**
- 🔥 2022.7.13：PP-Human v2 launched with a full upgrade of four industrial features: behavior analysis, attributes recognition, visitor traffic statistics and ReID. It provides a strong core algorithm for pedestrian detection, tracking and attribute analysis with a simple and detailed development process and model optimization strategy.
- 2022.4.18: Add  PP-Human practical tutorials, including training, deployment, and action expansion. Details for AIStudio project please see [Link](https://aistudio.baidu.com/aistudio/projectdetail/3842982)

- 2022.4.10: Add PP-Human examples; empower refined management of intelligent community management. A quick start for AIStudio [Link](https://aistudio.baidu.com/aistudio/projectdetail/3679564)
- 2022.4.5: Launch the real-time pedestrian analysis tool PP-Human. It supports pedestrian tracking, visitor traffic statistics, attributes recognition, and falling detection. Due to its specific optimization of real-scene data, it can accurately recognize various falling gestures, and adapt to different environmental backgrounds, light and camera angles.

![](https://user-images.githubusercontent.com/48054808/184843170-c3ef7d29-913b-4c6e-b533-b83892a8b0e2.gif)


## 🔮 Features and demonstration

### PP-Human

| ⭐ Feature                                          | 💟 Advantages                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | 💡Example                                                                                                                                     |
| -------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **ReID**                                           | Extraordinary performance: special optimization for technical challenges such as target occlusion, uncompleted and blurry objects to achieve mAP 98.8, 1.5ms/person                                                                                                                                                                                                                                                                                                                                                    | <img src="https://user-images.githubusercontent.com/48054808/173037607-0a5deadc-076e-4dcc-bd96-d54eea205f1f.png" title="" alt="" width="191"> |
| **Attribute analysis**                             | Compatible with a variety of data formats: support for images, video input<br/><br/>High performance: Integrated open-sourced datasets with real enterprise data for training, achieved mAP 94.86, 2ms/person<br/><br/>Support 26 attributes: gender, age, glasses, tops, shoes, hats, backpacks and other 26 high-frequency attributes                                                                                                                                                                                | <img src="https://user-images.githubusercontent.com/48054808/173036043-68b90df7-e95e-4ada-96ae-20f52bc98d7c.png" title="" alt="" width="207"> |
| **Behaviour detection**                            | Rich function: support five high-frequency anomaly behavior detection of falling, fighting, smoking, telephoning, and intrusion<br/><br/>Robust: unlimited by different environmental backgrounds, light, and camera angles.<br/><br/>High performance: Compared with video recognition technology, it takes significantly smaller computation resources; support localization and service-oriented rapid deployment<br/><br/>Fast training: only takes 15 minutes to produce high precision behavior detection models | <img src="https://user-images.githubusercontent.com/48054808/173034825-623e4f78-22a5-4f14-9b83-dc47aa868478.gif" title="" alt="" width="209"> |
| **Visitor traffic statistics**<br>**Trace record** | Simple and easy to use: single parameter to initiate functions of visitor traffic statistics and trace record                                                                                                                                                                                                                                                                                                                                                                                                          | <img src="https://user-images.githubusercontent.com/22989727/174736440-87cd5169-c939-48f8-90a1-0495a1fcb2b1.gif" title="" alt="" width="200"> |

### PP-Vehicle

| ⭐ Feature       | 💟 Advantages                                                                                    | 💡 Example                                                                                                                                         |
| ---------- | ------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **License Plate-Recognition**   | Both support for traditional plate and new green plate <br/><br/> Sample the frame in a time windows to recognice the plate license, and vots the license in many samples, which lead less compute cost and better accuracy, and the result is much more stable. <br/><br/>  hmean of text detector: 0.979;  accuracy of recognition model: 0.773   <br/><br/>                                  | <img title="" src="https://user-images.githubusercontent.com/48054808/185027987-6144cafd-0286-4c32-8425-7ab9515d1ec3.png" alt="" width="191"> |
| **Vehicle Attributes** | Identify 10 vehicle colors and 9 models <br/><br/> More powerfull backbone: PP-HGNet/PP-LCNet, with higher accuracy and faster speed <br/><br/> accuracy of model: 90.81 <br/><br/>  | <img title="" src="https://user-images.githubusercontent.com/48054808/185044490-00edd930-1885-4e79-b3d4-3a39a77dea93.gif" alt="" width="207"> |
| **Illegal Parking**   | Easy to use with one line command, and define the illegal area by yourself <br/><br/> Get the license of illegal car <br/><br/>  | <img title="" src="https://user-images.githubusercontent.com/48054808/185028419-58ae0af8-a035-42e7-9583-25f5e4ce0169.png" alt="" width="209"> |
| **in-out counting**  | Easy to use with one line command, and define the in-out line by yourself <br/><br/> Target route visualize with high tracking performance        | <img title="" src="https://user-images.githubusercontent.com/48054808/185028798-9e07379f-7486-4266-9d27-3aec943593e0.gif" alt="" width="200"> |
| **vehicle retrograde**   | Easy to use with one line command <br/><br/> High precision Segmetation model PP-LiteSeg    | <img title="" src="https://raw.githubusercontent.com/LokeZhou/PaddleDetection/develop/deploy/pipeline/docs/images/vehicle_retrograde.gif" alt="" width="200"> |
| **vehicle press line**  | Easy to use with one line command <br/><br/> High precision Segmetation model PP-LiteSeg    | <img title="" src="https://raw.githubusercontent.com/LokeZhou/PaddleDetection/develop/deploy/pipeline/docs/images/vehicle_press.gif" alt="" width="200"> |

## 🗳 Model Zoo

<details>
<summary><b>PP-Human End-to-end model results (click to expand)</b></summary>

| Task                                   | End-to-End Speed（ms） | Model                                                                                                                                                                                                                                                                                                                           | Size                                                                                                   |
|:--------------------------------------:|:--------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------:|
| Pedestrian detection (high precision)  | 25.1ms               | [Multi-object tracking](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)                                                                                                                                                                                                                      | 182M                                                                                                   |
| Pedestrian detection (lightweight)     | 16.2ms               | [Multi-object tracking](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_s_36e_pipeline.zip)                                                                                                                                                                                                                      | 27M                                                                                                    |
| Pedestrian detection (super lightweight) | 10ms(Jetson AGX)    | [Multi-object tracking](https://bj.bcebos.com/v1/paddledet/models/pipeline/pphuman/ppyoloe_plus_crn_t_auxhead_320_60e_pphuman.tar.gz)                                                                        | 17M                                         |
| Pedestrian tracking (high precision)   | 31.8ms               | [Multi-object tracking](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)                                                                                                                                                                                                                      | 182M                                                                                                   |
| Pedestrian tracking (lightweight)      | 21.0ms               | [Multi-object tracking](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_s_36e_pipeline.zip)                                                                                                                                                                                                                      | 27M                                                                                                    |
| Pedestrian tracking（super lightweight） | 13.2ms(Jetson AGX)    | [Multi-object tracking](https://bj.bcebos.com/v1/paddledet/models/pipeline/pphuman/ppyoloe_plus_crn_t_auxhead_320_60e_pphuman.tar.gz)                                                                        | 17M                                         |
|  MTMCT(REID)  |  Single Person 1.5ms | [REID](https://bj.bcebos.com/v1/paddledet/models/pipeline/reid_model.zip) | REID：92M |
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
| Vehicle detection (lightweight)     | 13.2ms               | [object detection](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_s_36e_ppvehicle.zip)                                                                                                                                                                                                                      | 27M                   |
| Vehicle detection (super lightweight)     | 10ms(Jetson AGX)               | [object detection](https://bj.bcebos.com/v1/paddledet/models/pipeline/ppvehicle/ppyoloe_plus_crn_t_auxhead_320_60e_ppvehicle.tar.gz)                                                                                                                                                                                                                      | 17M                                    |
| Vehicle tracking (high precision)   | 40ms               | [multi-object tracking](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_ppvehicle.zip)                                                                                                                                                                                                                      | 182M                                                                                                   |
| Vehicle tracking (lightweight)      | 25ms               | [multi-object tracking](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_s_36e_pipeline.zip)                                                                                                                                                                                                                      | 27M                                                                                                    |
| Vehicle tracking (super lightweight)     | 13.2ms(Jetson AGX)               | [multi-object tracking](https://bj.bcebos.com/v1/paddledet/models/pipeline/ppvehicle/ppyoloe_plus_crn_t_auxhead_320_60e_ppvehicle.tar.gz)                                                                                                                                                                                                                      | 17M                                    |
| Plate Recognition                   | 4.68ms     | [plate detection](https://bj.bcebos.com/v1/paddledet/models/pipeline/ch_PP-OCRv3_det_infer.tar.gz)<br>[plate recognition](https://bj.bcebos.com/v1/paddledet/models/pipeline/ch_PP-OCRv3_rec_infer.tar.gz)                                                                                         | Plate detection：3.9M<br>Plate recognition：12M                                    |
| Vehicle attribute      | 7.31ms               | [attribute recognition](https://bj.bcebos.com/v1/paddledet/models/pipeline/vehicle_attribute_model.zip)                                                                                                                                                                                                                      | 7.2M                                                                                        |
| Lane line Segmentation      |   47ms | [Lane line Segmentation](https://bj.bcebos.com/v1/paddledet/models/pipeline/pp_lite_stdc2_bdd100k.zip) | 47M |

</details>


Click to download the model, then unzip and save it in the `. /output_inference`.

## 📚 Doc Tutorials

### 🚶‍♀️ PP-Human

#### [A Quick Start](docs/tutorials/PPHuman_QUICK_STARTED_en.md)

#### Pedestrian attribute/feature recognition

* [A quick start](docs/tutorials/pphuman_attribute_en.md)

* [Customized development tutorials](../../docs/advanced_tutorials/customization/pphuman_attribute_en.md)

#### Behavior detection

* [A quick start](docs/tutorials/pphuman_action_en.md)

* [Customized development tutorials](../../docs/advanced_tutorials/customization/action_recognotion/README_en.md)

#### ReID

* [A quick start](docs/tutorials/pphuman_mtmct_en.md)

* [Customized development tutorials](../../docs/advanced_tutorials/customization/pphuman_mtmct_en.md)

#### Pedestrian tracking, visitor traffic statistics, trace records

* [A quick start](docs/tutorials/pphuman_mot_en.md)

* [Customized development tutorials](../../docs/advanced_tutorials/customization/pphuman_mot_en.md)


### 🚘 PP-Vehicle

#### [A Quick Start](docs/tutorials/PPVehicle_QUICK_STARTED.md)

#### Vehicle Plate License

- [A quick start](docs/tutorials/ppvehicle_plate_en.md)

- [Customized development tutorials](../../docs/advanced_tutorials/customization/ppvehicle_plate.md)

#### Vehicle Attributes

- [A quick start](docs/tutorials/ppvehicle_attribute_en.md)

- [Customized development tutorials](../../docs/advanced_tutorials/customization/ppvehicle_attribute_en.md)

#### Illegal Parking

- [A quick start](docs/tutorials/ppvehicle_illegal_parking_en.md)

- [Customized development tutorials](../../docs/advanced_tutorials/customization/pphuman_mot_en.md)

#### Vehicle Tracking/in-out counint/Route Visualize

- [A quick start](docs/tutorials/ppvehicle_mot_en.md)

- [Customized development tutorials](../../docs/advanced_tutorials/customization/pphuman_mot_en.md)

#### Vehicle Press Line

- [A quick start](docs/tutorials/ppvehicle_press_en.md)

- [Customized development tutorials](../../docs/advanced_tutorials/customization/ppvehicle_violation_en.md)

#### Vehicle Retrograde

- [A quick start](docs/tutorials/ppvehicle_retrograde_en.md)

- [Customized development tutorials](../../docs/advanced_tutorials/customization/ppvehicle_violation_en.md)
