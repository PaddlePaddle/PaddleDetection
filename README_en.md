English | [简体中文](README_cn.md)

<div align="center">
<p align="center">
  <img src="https://user-images.githubusercontent.com/48054808/160532560-34cf7a1f-d950-435e-90d2-4b0a679e5119.png" align="middle" width = "800" />
</p>

****A High-Efficient Development Toolkit for Object Detection based on [PaddlePaddle](https://github.com/paddlepaddle/paddle).****

[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)
[![Version](https://img.shields.io/github/release/PaddlePaddle/PaddleDetection.svg)](https://github.com/PaddlePaddle/PaddleDetection/releases)
![python version](https://img.shields.io/badge/python-3.6+-orange.svg)
![support os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg)

</div>

## <img src="https://user-images.githubusercontent.com/48054808/157793354-6e7f381a-0aa6-4bb7-845c-9acf2ecc05c3.png" width="20"/> Latest News

- 🔥 **2022.3.24：PaddleDetection [release 2.4 version](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4)**

  - Release GPU SOTA object detection series models (s/m/l/x) [PP-YOLOE](configs/ppyoloe), achieving mAP as 51.4% on COCO test dataset and 78.1 FPS on Nvidia V100, supporting AMP training and its training speed is 33% faster than PP-YOLOv2.
  - Release enhanced models of [PP-PicoDet](configs/picodet), including PP-PicoDet-XS model with 0.7M parameters, its mAP promoted ~2% on COCO, inference speed accelerated 63% on CPU, and post-processing integrated into the network to optimize deployment pipeline.
  - Release real-time human analysis tool [PP-Human](deploy/pphuman), which is based on data from real-life situations, supporting pedestrian detection, attribute recognition, human tracking, multi-camera tracking, human statistics and action recognition.

- 2021.11.03: Release [release/2.3](https://github.com/PaddlePaddle/Paddleetection/tree/release/2.3) version. Release mobile object detection model ⚡[PP-PicoDet](configs/picodet), mobile keypoint detection model ⚡[PP-TinyPose](configs/keypoint/tiny_pose)，Real-time tracking system [PP-Tracking](deploy/pptracking). Release object detection models, including [Swin-Transformer](configs/faster_rcnn), [TOOD](configs/tood), [GFL](configs/gfl), release [Sniper](configs/sniper) tiny object detection models and optimized [PP-YOLO-EB](configs/ppyolo) model for EdgeBoard. Release mobile keypoint detection model [Lite HRNet](configs/keypoint).

- 2021.08.10: Release [release/2.2](https://github.com/PaddlePaddle/Paddleetection/tree/release/2.2) version. Release Transformer object detection models, including [DETR](configs/detr), [Deformable DETR](configs/deformable_detr), [Sparse RCNN](configs/sparse_rcnn). Release [keypoint detection](configs/keypoint) models, including DarkHRNet and model trained on MPII dataset. Release [head-tracking](configs/mot/headtracking21) and [vehicle-tracking](configs/mot/vehicle) multi-object tracking models.

- 2021.05.20: Release [release/2.1](https://github.com/PaddlePaddle/Paddleetection/tree/release/2.1) version. Release [Keypoint Detection](configs/keypoint), including HigherHRNet and HRNet, [Multi-Object Tracking](configs/mot), including DeepSORT，JDE and FairMOT. Release model compression for PPYOLO series models.Update documents such as [EXPORT ONNX MODEL](deploy/EXPORT_ONNX_MODEL.md).

## <img title="" src="https://user-images.githubusercontent.com/48054808/157795569-9fc77c85-732f-4870-9be0-99a7fe2cff27.png" alt="" width="20"> Introduction

PaddleDetection is an end-to-end object detection development kit based on PaddlePaddle, which implements varied mainstream object detection, instance segmentation, tracking and keypoint detection algorithms in modular designwhich with configurable modules such as network components, data augmentations and losses, and release many kinds SOTA industry practice models, integrates abilities of model compression and cross-platform high-performance deployment, aims to help developers in the whole end-to-end development in a faster and better way.

#### PaddleDetection provides image processing capabilities such as object detection, instance segmentation, multi-object tracking, keypoint detection and etc.

<div  align="center">
  <img src="docs/images/ppdet.gif" width="800"/>
</div>

#### PaddleDetection covers industrialization, smart city, security & protection, retail, medicare industry and etc.

<div  align="center">
  <img src="https://user-images.githubusercontent.com/48054808/157826886-2e101a71-25a2-42f5-bf5e-30a97be28f46.gif" width="800"/>
</div>

## <img src="https://user-images.githubusercontent.com/48054808/157799599-e6a66855-bac6-4e75-b9c0-96e13cb9612f.png" width="20"/> Features

- **Rich Models**

  PaddleDetection provides rich of models, including **250+ pre-trained models** such as **object detection**, **instance segmentation**, **face detection**, **keypoint detection**, **multi-object tracking** and etc, covering a variety of **global competition champion** schemes.

- **Highly Flexible**

  Components are designed to be modular. Model architectures, as well as data preprocess pipelines and optimization strategies, can be easily customized with simple configuration changes.

- **Production Ready**

  From data augmentation, constructing models, training, compression, depolyment, get through end to end, and complete support for multi-architecture, multi-device deployment for **cloud and edge device**.

- **High Performance**

  Based on the high performance core of PaddlePaddle, advantages of training speed and memory occupation are obvious. FP16 training and multi-machine training are supported as well.

## <img title="" src="https://user-images.githubusercontent.com/48054808/157800467-2a9946ad-30d1-49a9-b9db-ba33413d9c90.png" alt="" width="20"> Community

- If you have any problem or suggestion on PaddleDetection, please send us issues through [GitHub Issues](https://github.com/PaddlePaddle/PaddleDetection/issues).

- Welcome to Join PaddleDetection QQ Group and Wechat Group (reply "Det").

  <div align="center">
  <img src="https://user-images.githubusercontent.com/48054808/157800129-2f9a0b72-6bb8-4b10-8310-93ab1639253f.jpg"  width = "200" />  
  <img src="https://user-images.githubusercontent.com/48054808/160531099-9811bbe6-cfbb-47d5-8bdb-c2b40684d7dd.png"  width = "200" />  
  </div>

## <img src="https://user-images.githubusercontent.com/48054808/157827140-03ffaff7-7d14-48b4-9440-c38986ea378c.png" width="20"/> Overview of Kit Structures

<table align="center">
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
        <ul>
          <li><b>Object Detection</b></li>
          <ul>
            <li>Faster RCNN</li>
            <li>FPN</li>
            <li>Cascade-RCNN</li>
            <li>Libra RCNN</li>
            <li>Hybrid Task RCNN</li>
            <li>PSS-Det</li>
            <li>RetinaNet</li>
            <li>YOLOv3</li>
            <li>YOLOv4</li>  
            <li>PP-YOLOv1/v2</li>
            <li>PP-YOLO-Tiny</li>
            <li>SSD</li>
            <li>CornerNet-Squeeze</li>
            <li>FCOS</li>  
            <li>TTFNet</li>
            <li>PP-PicoDet</li>
            <li>DETR</li>
            <li>Deformable DETR</li>
            <li>Swin Transformer</li>
            <li>Sparse RCNN</li>
        </ul>
        <li><b>Instance Segmentation</b></li>
        <ul>
            <li>Mask RCNN</li>
            <li>SOLOv2</li>
        </ul>
        <li><b>Face Detection</b></li>
        <ul>
            <li>FaceBoxes</li>
            <li>BlazeFace</li>
            <li>BlazeFace-NAS</li>
        </ul>
        <li><b>Multi-Object-Tracking</b></li>
        <ul>
            <li>JDE</li>
            <li>FairMOT</li>
            <li>DeepSort</li>
        </ul>
        <li><b>KeyPoint-Detection</b></li>
        <ul>
            <li>HRNet</li>
            <li>HigherHRNet</li>
        </ul>
      </ul>
      </td>
      <td>
        <ul>
          <li>ResNet(&vd)</li>
          <li>ResNeXt(&vd)</li>
          <li>SENet</li>
          <li>Res2Net</li>
          <li>HRNet</li>
          <li>Hourglass</li>
          <li>CBNet</li>
          <li>GCNet</li>
          <li>DarkNet</li>
          <li>CSPDarkNet</li>
          <li>VGG</li>
          <li>MobileNetv1/v3</li>  
          <li>GhostNet</li>
          <li>Efficientnet</li>  
          <li>BlazeNet</li>  
        </ul>
      </td>
      <td>
        <ul><li><b>Common</b></li>
          <ul>
            <li>Sync-BN</li>
            <li>Group Norm</li>
            <li>DCNv2</li>
            <li>Non-local</li>
          </ul>  
        </ul>
        <ul><li><b>KeyPoint</b></li>
          <ul>
            <li>DarkPose</li>
          </ul>  
        </ul>
        <ul><li><b>FPN</b></li>
          <ul>
            <li>BiFPN</li>
            <li>BFP</li>  
            <li>HRFPN</li>
            <li>ACFPN</li>
          </ul>  
        </ul>  
        <ul><li><b>Loss</b></li>
          <ul>
            <li>Smooth-L1</li>
            <li>GIoU/DIoU/CIoU</li>  
            <li>IoUAware</li>
          </ul>  
        </ul>  
        <ul><li><b>Post-processing</b></li>
          <ul>
            <li>SoftNMS</li>
            <li>MatrixNMS</li>  
          </ul>  
        </ul>
        <ul><li><b>Speed</b></li>
          <ul>
            <li>FP16 training</li>
            <li>Multi-machine training </li>  
          </ul>  
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

## <img src="https://user-images.githubusercontent.com/48054808/157801371-9a9a8c65-1690-4123-985a-e0559a7f9494.png" width="20"/> Overview of Model Performance

The relationship between COCO mAP and FPS on Tesla V100 of representative models of each server side architectures and backbones.

<div align="center">
  <img src="docs/images/fps_map.png" />
</div>

**NOTE:**

- `CBResNet stands` for `Cascade-Faster-RCNN-CBResNet200vd-FPN`, which has highest mAP on COCO as 53.3%

- `Cascade-Faster-RCNN` stands for `Cascade-Faster-RCNN-ResNet50vd-DCN`, which has been optimized to 20 FPS inference speed when COCO mAP as 47.8% in PaddleDetection models

- `PP-YOLO` achieves mAP of 45.9% on COCO and 72.9FPS on Tesla V100. Both precision and speed surpass [YOLOv4](https://arxiv.org/abs/2004.10934)

- `PP-YOLO v2` is optimized version of `PP-YOLO` which has mAP of 49.5% and 68.9FPS on Tesla V100
- `PP-YOLOE` is optimized version of `PP-YOLO v2` which has mAP of 51.4% and 78.1FPS on Tesla V100
- All these models can be get in [Model Zoo](#ModelZoo)

The relationship between COCO mAP and FPS on Qualcomm Snapdragon 865 of representative mobile side models.

<div align="center">
  <img src="docs/images/mobile_fps_map.png" width=600/>
</div>

**NOTE:**

- All data tested on Qualcomm Snapdragon 865(4*A77 + 4*A55) processor with batch size of 1 and CPU threads of 4, and use NCNN library in testing, benchmark scripts is publiced at [MobileDetBenchmark](https://github.com/JiweiMaster/MobileDetBenchmark)
- [PP-PicoDet](configs/picodet) and [PP-YOLO-Tiny](configs/ppyolo) are developed and released by PaddleDetection, other models are not provided in PaddleDetection.

## <img src="https://user-images.githubusercontent.com/48054808/157828296-d5eb0ccb-23ea-40f5-9957-29853d7d13a9.png" width="20"/> Tutorials

### Get Started

- [Installation guide](docs/tutorials/INSTALL.md)
- [Prepare dataset](docs/tutorials/PrepareDataSet_en.md)
- [Quick start on PaddleDetection](docs/tutorials/GETTING_STARTED.md)

### Advanced Tutorials

- Parameter configuration

  - [Parameter configuration for RCNN model](docs/tutorials/config_annotation/faster_rcnn_r50_fpn_1x_coco_annotation_en.md)
  - [Parameter configuration for PP-YOLO model](docs/tutorials/config_annotation/ppyolo_r50vd_dcn_1x_coco_annotation_en.md)

- Model Compression(Based on [PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim))

  - [Prune/Quant/Distill](configs/slim)

- Inference and deployment

  - [Export model for inference](deploy/EXPORT_MODEL_en.md)
  - [Paddle Inference](deploy/README_en.md)
    - [Python inference](deploy/python)
    - [C++ inference](deploy/cpp)
  - [Paddle-Lite](deploy/lite)
  - [Paddle Serving](deploy/serving)
  - [Export ONNX model](deploy/EXPORT_ONNX_MODEL_en.md)
  - [Inference benchmark](deploy/BENCHMARK_INFER_en.md)
  - [Exporting to ONNX and using OpenVINO for inference](docs/advanced_tutorials/openvino_inference/README.md)

- Advanced development

  - [New data augmentations](docs/advanced_tutorials/READER_en.md)
  - [New detection algorithms](docs/advanced_tutorials/MODEL_TECHNICAL.md)

## <img src="https://user-images.githubusercontent.com/48054808/157829890-a535b8a6-631c-4c87-b861-64d4b32b2d6a.png" width="20"/> Model Zoo

- Universal object detection
  - [Model library and baselines](docs/MODEL_ZOO_cn.md)
  - [PP-YOLOE](configs/ppyoloe/README_cn.md)
  - [PP-YOLO](configs/ppyolo/README.md)
  - [PP-PicoDet](configs/picodet/README.md)
  - [Enhanced Anchor Free model--TTFNet](configs/ttfnet/README_en.md)
  - [Mobile models](static/configs/mobile/README_en.md)
  - [676 classes of object detection](static/docs/featured_model/LARGE_SCALE_DET_MODEL_en.md)
  - [Two-stage practical PSS-Det](configs/rcnn_enhance/README_en.md)
  - [SSLD pretrained models](docs/feature_models/SSLD_PRETRAINED_MODEL_en.md)
- Universal instance segmentation
  - [SOLOv2](configs/solov2/README.md)
- Rotation object detection
  - [S2ANet](configs/dota/README_en.md)
- [Keypoint detection](configs/keypoint)
  - [PP-TinyPose](configs/keypoint/tiny_pose)
  - HigherHRNet
  - HRNet
  - LiteHRNet
- [Multi-Object Tracking](configs/mot/README.md)
  - [PP-Tracking](deploy/pptracking/README.md)
  - [DeepSORT](configs/mot/deepsort/README.md)
  - [JDE](configs/mot/jde/README.md)
  - [FairMOT](configs/mot/fairmot/README.md)
- Vertical field
  - [Face detection](configs/face_detection/README_en.md)
  - [Pedestrian detection](configs/pedestrian/README.md)
  - [Vehicle detection](configs/vehicle/README.md)
  - [Real-Time Human Analysis Tool PP-Human](deploy/pphuman)
- Competition Plan
  - [Objects365 2019 Challenge champion model](static/docs/featured_model/champion_model/CACascadeRCNN_en.md)
  - [Best single model of Open Images 2019-Object Detection](static/docs/featured_model/champion_model/OIDV5_BASELINE_MODEL_en.md)

## <img title="" src="https://user-images.githubusercontent.com/48054808/157836473-1cf451fa-f01f-4148-ba68-b6d06d5da2f9.png" alt="" width="20"> Applications

- [Christmas portrait automatic generation tool](static/application/christmas)
- [Android Fitness Demo](https://github.com/zhiboniu/pose_demo_android)

## <img src="https://user-images.githubusercontent.com/48054808/157835981-ef6057b4-6347-4768-8fcc-cd07fcc3d8b0.png" width="20"/> Updates

For the details of version update, please refer to [Version Update Doc](docs/CHANGELOG.md).

## <img title="" src="https://user-images.githubusercontent.com/48054808/157835345-f5d24128-abaf-4813-b793-d2e5bdc70e5a.png" alt="" width="20"> License

PaddleDetection is released under the [Apache 2.0 license](LICENSE).

## <img src="https://user-images.githubusercontent.com/48054808/157835796-08d4ffbc-87d9-4622-89d8-cf11a44260fc.png" width="20"/> Contribution

Contributions are highly welcomed and we would really appreciate your feedback!!

- Thanks [Mandroide](https://github.com/Mandroide) for cleaning the code and unifying some function interface.
- Thanks [FL77N](https://github.com/FL77N/) for contributing the code of `Sparse-RCNN` model.
- Thanks [Chen-Song](https://github.com/Chen-Song) for contributing the code of `Swin Faster-RCNN` model.
- Thanks [yangyudong](https://github.com/yangyudong2020), [hchhtc123](https://github.com/hchhtc123) for contributing PP-Tracking GUI interface.
- Thanks [Shigure19](https://github.com/Shigure19) for contributing PP-TinyPose fitness APP.

## <img src="https://user-images.githubusercontent.com/48054808/157835276-9aab9d1c-1c46-446b-bdd4-5ab75c5cfa48.png" width="20"/> Citation

```
@misc{ppdet2019,
title={PaddleDetection, Object detection and instance segmentation toolkit based on PaddlePaddle.},
author={PaddlePaddle Authors},
howpublished = {\url{https://github.com/PaddlePaddle/PaddleDetection}},
year={2019}
}
```
