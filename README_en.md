English | [简体中文](README_cn.md)


# Product news

- 2021.11.03: Release [release/2.3](https://github.com/PaddlePaddle/Paddleetection/tree/release/2.3) version. Release mobile object detection model ⚡[PP-PicoDet](configs/picodet), mobile keypoint detection model ⚡[PP-TinyPose](configs/keypoint/tiny_pose)，Real-time tracking system [PP-Tracking](deploy/pptracking). Release object detection models, including [Swin-Transformer](configs/faster_rcnn), [TOOD](configs/tood), [GFL](configs/gfl), release [Sniper](configs/sniper) tiny object detection models and optimized [PP-YOLO-EB](configs/ppyolo) model for EdgeBoard. Release mobile keypoint detection model [Lite HRNet](configs/keypoint).
- 2021.08.10: Release [release/2.2](https://github.com/PaddlePaddle/Paddleetection/tree/release/2.2) version. Release Transformer object detection models, including [DETR](configs/detr), [Deformable DETR](configs/deformable_detr), [Sparse RCNN](configs/sparse_rcnn). Release [keypoint detection](configs/keypoint) models, including DarkHRNet and model trained on MPII dataset. Release [head-tracking](configs/mot/headtracking21) and [vehicle-tracking](configs/mot/vehicle) multi-object tracking models.
- 2021.05.20: Release [release/2.1](https://github.com/PaddlePaddle/Paddleetection/tree/release/2.1) version. Release [Keypoint Detection](configs/keypoint), including HigherHRNet and HRNet, [Multi-Object Tracking](configs/mot), including DeepSORT，JDE and FairMOT. Release model compression for PPYOLO series models.Update documents such as [EXPORT ONNX MODEL](deploy/EXPORT_ONNX_MODEL.md).


# Introduction

PaddleDetection is an end-to-end object detection development kit based on PaddlePaddle, which implements varied mainstream object detection, instance segmentation, tracking and keypoint detection algorithms in modular designwhich with configurable modules such as network components, data augmentations and losses, and release many kinds SOTA industry practice models, integrates abilities of model compression and cross-platform high-performance deployment, aims to help developers in the whole end-to-end development in a faster and better way.

### PaddleDetection provides image processing capabilities such as object detection, instance segmentation, multi-object tracking, keypoint detection and etc.

<div width="1000" align="center">
  <img src="docs/images/ppdet.gif"/>
</div>


### Features

- **Rich Models**
PaddleDetection provides rich of models, including **100+ pre-trained models** such as **object detection**, **instance segmentation**, **face detection** etc. It covers a variety of **global competition champion** schemes.

- **Highly Flexible:**
Components are designed to be modular. Model architectures, as well as data preprocess pipelines and optimization strategies, can be easily customized with simple configuration changes.

- **Production Ready:**
From data augmentation, constructing models, training, compression, depolyment, get through end to end, and complete support for multi-architecture, multi-device deployment for **cloud and edge device**.

- **High Performance:**
Based on the high performance core of PaddlePaddle, advantages of training speed and memory occupation are obvious. FP16 training and multi-machine training are supported as well.

#### Overview of Kit Structures

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

#### Overview of Model Performance

The relationship between COCO mAP and FPS on Tesla V100 of representative models of each server side architectures and backbones.

<div align="center">
  <img src="docs/images/fps_map.png" />
  </div>

  **NOTE:**

  - `CBResNet stands` for `Cascade-Faster-RCNN-CBResNet200vd-FPN`, which has highest mAP on COCO as 53.3%

  - `Cascade-Faster-RCNN` stands for `Cascade-Faster-RCNN-ResNet50vd-DCN`, which has been optimized to 20 FPS inference speed when COCO mAP as 47.8% in PaddleDetection models

  - `PP-YOLO` achieves mAP of 45.9% on COCO and 72.9FPS on Tesla V100. Both precision and speed surpass [YOLOv4](https://arxiv.org/abs/2004.10934)

  - `PP-YOLO v2` is optimized version of `PP-YOLO` which has mAP of 49.5% and 68.9FPS on Tesla V100

  - All these models can be get in [Model Zoo](#ModelZoo)

The relationship between COCO mAP and FPS on Qualcomm Snapdragon 865 of representative mobile side models.

<div align="center">
  <img src="docs/images/mobile_fps_map.png" width=600 />
</div>

**NOTE:**

- All data tested on Qualcomm Snapdragon 865(4\*A77 + 4\*A55) processor with batch size of 1 and CPU threads of 4, and use NCNN library in testing, benchmark scripts is publiced at [MobileDetBenchmark](https://github.com/JiweiMaster/MobileDetBenchmark)
- [PP-PicoDet](configs/picodet) and [PP-YOLO-Tiny](configs/ppyolo) are developed and released by PaddleDetection, other models are not provided in PaddleDetection.

## Tutorials

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
  - [Exporting FairMOT to ONNX and using OpenVINO for inference](docs/advanced_tutorials/fairmot/OPENVINO_INFERENCE.md)

- Advanced development
  - [New data augmentations](docs/advanced_tutorials/READER_en.md)
  - [New detection algorithms](docs/advanced_tutorials/MODEL_TECHNICAL_en.md)


## Model Zoo

- Universal object detection
  - [Model library and baselines](docs/MODEL_ZOO_cn.md)
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
- Competition Plan
  - [Objects365 2019 Challenge champion model](static/docs/featured_model/champion_model/CACascadeRCNN_en.md)
  - [Best single model of Open Images 2019-Object Detection](static/docs/featured_model/champion_model/OIDV5_BASELINE_MODEL_en.md)

## Applications

- [Christmas portrait automatic generation tool](static/application/christmas)
- [Android Fitness Demo](https://github.com/zhiboniu/pose_demo_android)

## Updates

Updates please refer to [change log](docs/CHANGELOG_en.md) for details.


## License

PaddleDetection is released under the [Apache 2.0 license](LICENSE).


## Contributing

Contributions are highly welcomed and we would really appreciate your feedback!!
- Thanks [Mandroide](https://github.com/Mandroide) for cleaning the code and unifying some function interface.
- Thanks [FL77N](https://github.com/FL77N/) for contributing the code of `Sparse-RCNN` model.
- Thanks [Chen-Song](https://github.com/Chen-Song) for contributing the code of `Swin Faster-RCNN` model.
- Thanks [yangyudong](https://github.com/yangyudong2020), [hchhtc123](https://github.com/hchhtc123) for contributing PP-Tracking GUI interface.
- Thanks [Shigure19](https://github.com/Shigure19) for contributing PP-TinyPose fitness APP.

## Citation

```
@misc{ppdet2019,
title={PaddleDetection, Object detection and instance segmentation toolkit based on PaddlePaddle.},
author={PaddlePaddle Authors},
howpublished = {\url{https://github.com/PaddlePaddle/PaddleDetection}},
year={2019}
}
```
