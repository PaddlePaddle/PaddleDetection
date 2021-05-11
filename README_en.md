English | [简体中文](README_cn.md)

### PaddleDetection 2.0 is ready! Dygraph mode is set by default and static graph code base is [here](static)

### Highly effective PPYOLO v2 and ultra lightweight PPYOLO tiny are released! [link](configs/ppyolo/README.md)

### SOTA Anchor Free model -- PAFNet is released! [link](configs/ttfnet/README.md)
# Recent Activity

A series of live lectures on detailed explanation of industrial-level object detection technology is about to be launched，to see how powerful PP-YOLOv2 surpasses YOLOv5.

Welcome to the PPYLOv2 &Tiny Tech Seminar Group


<div align="left">
  <img src="https://z3.ax1x.com/2021/05/11/gUDw0e.png" width='150'/>
</div>


### Course Schedule
[Live Link](http://live.bilibili.com/21689802)
* May 13rd 19:00-20:00
  -  Topic: Detailed Interpretation of Industrial-level Object Detection Algorithms
* May 14th 19:00-20:00
   - Topic: Analysis and Application of 1.3M Ultra-lightweight Object Detection Algorithm 
* May 21st 20:00-21:00
   - Topic: The Model Development Exercise of Small Target Detection Under Complex Background

### Course Link

[0【PaddleDetection2.0 Special】Quick Experience of New Version](https://aistudio.baidu.com/aistudio/projectdetail/1885319)

 [1【PaddleDetection2.0 Special】How to Customize Dataset](https://aistudio.baidu.com/aistudio/projectdetail/1917140)

 [2【PaddleDetection2.0 Special】Quick Start PP-YOLOv2](https://aistudio.baidu.com/aistudio/projectdetail/1922155)

 [3【PaddleDetection2.0 Special】Quick Start PP-YOLO tiny](https://aistudio.baidu.com/aistudio/projectdetail/1918450)

 [4【PaddleDetection2.0 Special】Quick Start S2ANet](https://aistudio.baidu.com/aistudio/projectdetail/1923957)

 [5【PaddleDetection2.0 Special】Fast Implementation of Pedestrian Detection](https://aistudio.baidu.com/aistudio/projectdetail/1918451)

 [6【PaddleDetection2.0 Special】Fast Implementation of Face Detection](https://aistudio.baidu.com/aistudio/projectdetail/1918453)
 


# Introduction

PaddleDetection is an end-to-end object detection development kit based on PaddlePaddle, which aims to help developers in the whole development of constructing, training, optimizing and deploying detection models in a faster and better way.

PaddleDetection implements varied mainstream object detection algorithms in modular design, and provides wealthy data augmentation methods, network components(such as backbones), loss functions, etc., and integrates abilities of model compression and cross-platform high-performance deployment.

After a long time of industry practice polishing, PaddleDetection has had smooth and excellent user experience, it has been widely used by developers in more than ten industries such as industrial quality inspection, remote sensing image object detection, automatic inspection, new retail, Internet, and scientific research.

<div align="center">
  <img src="static/docs/images/football.gif" width='800'/>
</div>

### Product news

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
- [Quick start on small dataset](docs/tutorials/QUICK_STARTED_en.md)
- [Prepare dataset](docs/tutorials/PrepareDataSet.md)
- [Train/Evaluation/Inference/Deploy](docs/tutorials/GETTING_STARTED_en.md)


### Advanced Tutorials

- Parameter configuration
  - [Parameter configuration for RCNN model](docs/tutorials/config_annotation/faster_rcnn_r50_fpn_1x_coco_annotation.md)
  - [Parameter configuration for PP-YOLO model](docs/tutorials/config_annotation/ppyolo_r50vd_dcn_1x_coco_annotation.md)

- Model Compression(Based on [PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim))
  - [Prune/Quant/Distill](configs/slim)

- Inference and deployment
  - [Export model for inference](deploy/EXPORT_MODEL.md)
  - [Python inference](deploy/python)
  - [C++ inference](deploy/cpp)
  - [Serving](deploy/serving)
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
