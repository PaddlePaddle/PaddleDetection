English | [简体中文](README_cn.md)

### PaddleDetection 2.0 is ready! Dygraph mode is set by default and static graph code base is [here](static)

### [Keypoint detection](configs/keypoint) and [Multi-Object Tracking](configs/mot) are released!

### Highly effective PPYOLO v2 and ultra lightweight PPYOLO tiny are released! [link](configs/ppyolo/README.md)

### SOTA Anchor Free model -- PAFNet is released! [link](configs/ttfnet/README.md)

# Introduction

PaddleDetection is an end-to-end object detection development kit based on PaddlePaddle, which aims to help developers in the whole development of constructing, training, optimizing and deploying detection models in a faster and better way.

PaddleDetection implements varied mainstream object detection algorithms in modular design, and provides wealthy data augmentation methods, network components(such as backbones), loss functions, etc., and integrates abilities of model compression and cross-platform high-performance deployment.

After a long time of industry practice polishing, PaddleDetection has had smooth and excellent user experience, it has been widely used by developers in more than ten industries such as industrial quality inspection, remote sensing image object detection, automatic inspection, new retail, Internet, and scientific research.

<div align="center">
  <img src="static/docs/images/football.gif" width='800'/>
  <img src="docs/images/mot_pose_demo_640x360.gif" width='800'/>
</div>

### Product news

- 2021.05.20: Release `release/2.1` version. Release [Keypoint Detection](configs/keypoint), including HigherHRNet and HRNet, [Multi-Object Tracking](configs/mot), including DeepSORT，JDE and FairMOT. Release model compression for PPYOLO series models.Update documents such as [EXPORT ONNX MODEL](deploy/EXPORT_ONNX_MODEL.md). Please refer to [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.1) for details.
- 2021.04.14: Release `release/2.0` version. Dygraph mode in PaddleDetection is fully supported. Cover all the algorithm of static graph and update the performance of mainstream detection models. Release [`PP-YOLO v2` and `PP-YOLO tiny`](configs/ppyolo/README.md), enhanced anchor free model [PAFNet](configs/ttfnet/README.md) and [`S2ANet`](configs/dota/README.md) which is aimed at rotation object detection.Please refer to [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.0) for details.
- 2020.02.07: Release `release/2.0-rc` version, Please refer to [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.0-rc) for details.


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

<table>
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
        <ul><li><b>Two-Stage Detection</b></li>
          <ul>
            <li>Faster RCNN</li>
            <li>FPN</li>
            <li>Cascade-RCNN</li>
            <li>Libra RCNN</li>
            <li>Hybrid Task RCNN</li>
            <li>PSS-Det RCNN</li>
          </ul>
        </ul>
        <ul><li><b>One-Stage Detection</b></li>
          <ul>
            <li>RetinaNet</li>
            <li>YOLOv3</li>
            <li>YOLOv4</li>  
            <li>PP-YOLO</li>
            <li>SSD</li>
          </ul>
        </ul>
        <ul><li><b>Anchor Free</b></li>
          <ul>
            <li>CornerNet-Squeeze</li>
            <li>FCOS</li>  
            <li>TTFNet</li>
          </ul>
        </ul>
        <ul>
          <li><b>Instance Segmentation</b></li>
            <ul>
             <li>Mask RCNN</li>
             <li>SOLOv2</li>
            </ul>
        </ul>
        <ul>
          <li><b>Face-Detction</b></li>
            <ul>
             <li>FaceBoxes</li>
             <li>BlazeFace</li>
             <li>BlazeFace-NAS</li>
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
          <li>Flipping</li>  
          <li>Expand</li>
          <li>Crop</li>
          <li>Color Distort</li>  
          <li>Random Erasing</li>  
          <li>Mixup </li>
          <li>Cutmix </li>
          <li>Grid Mask</li>
          <li>Auto Augment</li>  
        </ul>  
      </td>  
    </tr>


</td>
    </tr>
  </tbody>
</table>

#### Overview of Model Performance
The relationship between COCO mAP and FPS on Tesla V100 of representative models of each architectures and backbones.

<div align="center">
  <img src="docs/images/fps_map.png" />
</div>

**NOTE:**

- `CBResNet stands` for `Cascade-Faster-RCNN-CBResNet200vd-FPN`, which has highest mAP on COCO as 53.3%

- `Cascade-Faster-RCNN` stands for `Cascade-Faster-RCNN-ResNet50vd-DCN`, which has been optimized to 20 FPS inference speed when COCO mAP as 47.8% in PaddleDetection models

- `PP-YOLO` achieves mAP of 45.9% on COCO and 72.9FPS on Tesla V100. Both precision and speed surpass [YOLOv4](https://arxiv.org/abs/2004.10934)

- `PP-YOLO v2` is optimized version of `PP-YOLO` which has mAP of 49.5% and 68.9FPS on Tesla V100

- All these models can be get in [Model Zoo](#ModelZoo)


## Tutorials

### Get Started

- [Installation guide](docs/tutorials/INSTALL_en.md)
- [Prepare dataset](docs/tutorials/PrepareDataSet.md)
- [Quick start on PaddleDetection](docs/tutorials/GETTING_STARTED_cn.md)


### Advanced Tutorials

- Parameter configuration
  - [Parameter configuration for RCNN model](docs/tutorials/config_annotation/faster_rcnn_r50_fpn_1x_coco_annotation.md)
  - [Parameter configuration for PP-YOLO model](docs/tutorials/config_annotation/ppyolo_r50vd_dcn_1x_coco_annotation.md)

- Model Compression(Based on [PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim))
  - [Prune/Quant/Distill](configs/slim)

- Inference and deployment
  - [Export model for inference](deploy/EXPORT_MODEL.md)
  - [Paddle Inference](deploy/README.md)
      - [Python inference](deploy/python)
      - [C++ inference](deploy/cpp)
  - [Paddle-Lite](deploy/lite)
  - [Paddle Serving](deploy/serving)
  - [Export ONNX model](deploy/EXPORT_ONNX_MODEL.md)
  - [Inference benchmark](deploy/BENCHMARK_INFER.md)

- Advanced development
  - [New data augmentations](docs/advanced_tutorials/READER.md)
  - [New detection algorithms](docs/advanced_tutorials/MODEL_TECHNICAL.md)


## Model Zoo

- Universal object detection
  - [Model library and baselines](docs/MODEL_ZOO_cn.md)
  - [PP-YOLO](configs/ppyolo/README.md)
  - [Enhanced Anchor Free model--TTFNet](configs/ttfnet/README.md)
  - [Mobile models](static/configs/mobile/README.md)
  - [676 classes of object detection](static/docs/featured_model/LARGE_SCALE_DET_MODEL.md)
  - [Two-stage practical PSS-Det](configs/rcnn_enhance/README.md)
  - [SSLD pretrained models](docs/feature_models/SSLD_PRETRAINED_MODEL_en.md)
- Universal instance segmentation
  - [SOLOv2](configs/solov2/README.md)
- Rotation object detection
  - [S2ANet](configs/dota/README.md)
- [Keypoint detection](configs/keypoint)
  - HigherHRNet
  - HRNeet
- [Multi-Object Tracking](configs/mot/README.md)
  - [DeepSORT](configs/mot/deepsort/README.md)
  - [JDE](configs/mot/jde/README.md)
  - [FairMOT](configs/mot/fairmot/README.md)
- Vertical field
  - [Face detection](configs/face_detection/README.md)
  - [Pedestrian detection](configs/pedestrian/README.md)
  - [Vehicle detection](configs/vehicle/README.md)
- Competition Plan
  - [Objects365 2019 Challenge champion model](static/docs/featured_model/champion_model/CACascadeRCNN.md)
  - [Best single model of Open Images 2019-Object Detction](static/docs/featured_model/champion_model/OIDV5_BASELINE_MODEL.md)

## Applications

- [Christmas portrait automatic generation tool](static/application/christmas)

## Updates

v2.1 was released at `05/2021`, Release Keypoint Detection and Multi-Object Tracking. Release model compression for PPYOLO series. Update documents such as export ONNX model. Please refer to [change log](docs/CHANGELOG.md) for details.

v2.0 was released at `04/2021`, fully support dygraph version, which add BlazeFace, PSS-Det and plenty backbones, release `PP-YOLOv2`, `PP-YOLO tiny` and `S2ANet`, support model distillation and VisualDL, add inference benchmark, etc. Please refer to [change log](docs/CHANGELOG.md) for details.


## License

PaddleDetection is released under the [Apache 2.0 license](LICENSE).


## Contributing

Contributions are highly welcomed and we would really appreciate your feedback!!

## Citation

```
@misc{ppdet2019,
title={PaddleDetection, Object detection and instance segmentation toolkit based on PaddlePaddle.},
author={PaddlePaddle Authors},
howpublished = {\url{https://github.com/PaddlePaddle/PaddleDetection}},
year={2019}
}
```
