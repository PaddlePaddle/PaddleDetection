English | [简体中文](README_cn.md)

Documentation:[https://paddledetection.readthedocs.io](https://paddledetection.readthedocs.io)

# Introduction

PaddleDetection is an end-to-end object detection development kit based on PaddlePaddle, which aims to help developers in the whole development of constructing, training, optimizing and deploying detection models in a faster and better way.

PaddleDetection implements varied mainstream object detection algorithms in modular design, and provides wealthy data augmentation methods, network components(such as backbones), loss functions, etc., and integrates abilities of model compression and cross-platform high-performance deployment.

After a long time of industry practice polishing, PaddleDetection has had smooth and excellent user experience, it has been widely used by developers in more than ten industries such as industrial quality inspection, remote sensing image object detection, automatic inspection, new retail, Internet, and scientific research.

<div align="center">
  <img src="docs/images/football.gif" width='800'/>
</div>

### Product dynamic

- 2020.11.20: Release `release/0.5` version, Please refer to [change log](docs/CHANGELOG.md) for details.
- 2020.11.10: Added [SOLOv2](configs/solov2) as an instance segmentation model, which reached 38.6 FPS on a single Tesla V100, 38.8 mask AP on Coco-Val dataset, and  inference speed increased by 24% and mAP by 2.4 percentage points.
- 2020.10.30: PP-YOLO support rectangular image input, and add a new PACT quantization strategy for slim。
- 2020.09.30: Released the [mobile-side detection demo](deploy/android_demo), and you can directly scan the code for installation experience.
- 2020.09.21-27: [Object detection 7 days of punching class] Hand in hand to teach you from the beginning to the advanced level, in-depth understanding of the object detection algorithm life. Join the course QQ group (1136406895) to study together :)
- 2020.07.24: [PP-YOLO](https://arxiv.org/abs/2007.12099), which is **the most practical** object detection model, was released, it deeply considers the double demands of industrial applications for accuracy and speed, and reached accuracy as 45.2% (the latest 45.9%) on COCO dataset, inference speed as 72.9 FPS on a single Test V100. Please refer to [PP-YOLO](configs/ppyolo/README.md) for details.
- 2020.06.11: Publish 676 classes of large-scale server-side practical object detection models that are applicable to most application scenarios and can be used directly for prediction or for fine-tuning other tasks.

### Features

- **Rich Models**
PaddleDetection provides rich of models, including **100+ pre-trained models** such as **object detection**, **instance segmentation**, **face detection** etc. It covers a variety of **global competition champion** schemes.

- **Use Concisely**
Modular design, decouple each network component, developers easily build and try various detection models and optimization strategies, quickly get high-performance, customized algorithm.

- **Getting Through End to End**
From data augmentation, constructing models, training, compression, depolyment, get through end to end, and complete support for multi-architecture, multi-device deployment for **cloud and edge device**.

- **High Performance:**
Based on the high performance core of PaddlePaddle, advantages of training speed and memory occupation are obvious. Support FP16 training, support multi-machine training.

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
  <img src="docs/images/map_fps.png" />
</div>

**NOTE:**

- `CBResNet stands` for `Cascade-Faster-RCNN-CBResNet200vd-FPN`, which has highest mAP on COCO as 53.3%

- `Cascade-Faster-RCNN` stands for `Cascade-Faster-RCNN-ResNet50vd-DCN`, which has been optimized to 20 FPS inference speed when COCO mAP as 47.8% in PaddleDetection models

- The enhanced PaddleDetection model `YOLOv3-ResNet50vd-DCN` is 10.6 absolute percentage points higher than paper on COCO mAP, and inference speed is 61.3 fps, nearly 70% faster than the darknet framework.
All these models can be get in [Model Zoo](#ModelZoo)


## Tutorials

### Get Started

- [Installation guide](docs/tutorials/INSTALL_cn.md)
- [Quick start on small dataset](docs/tutorials/QUICK_STARTED_cn.md)
- [Prepare dataset](docs/tutorials/PrepareDataSet.md)
- [Train/Evaluation/Inference/Deploy](docs/tutorials/DetectionPipeline.md)
- [How to train a custom dataset](docs/tutorials/Custom_DataSet.md)
- [FAQ](docs/FAQ.md)

### Advanced Tutorials

- Parameter configuration
  - [Introduction to the configuration workflow](docs/advanced_tutorials/config_doc/CONFIG_cn.md)
  - [Parameter configuration for RCNN model](docs/advanced_tutorials/config_doc/RCNN_PARAMS_DOC.md)
  - [Parameter configuration for YOLOv3 model](docs/advanced_tutorials/config_doc/yolov3_mobilenet_v1.md)

- Tansfer learning
  - [How to load pretrained model](docs/advanced_tutorials/TRANSFER_LEARNING_cn.md)

- Model Compression(Based on [PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim))
  - [Model compression benchmark](slim)
  - [Quantization](slim/quantization)
  - [Model pruning](slim/prune)
  - [Model distillation](slim/distillation)
  - [Neural Architecture Search](slim/nas)

- Inference and deployment
  - [Export model for inference](docs/advanced_tutorials/deploy/EXPORT_MODEL.md)
  - [Python inference](deploy/python)
  - [C++ inference](deploy/cpp)
  - [Mobile](https://github.com/PaddlePaddle/Paddle-Lite-Demo)
  - [Serving](deploy/serving)
  - [Inference benchmark](docs/advanced_tutorials/deploy/BENCHMARK_INFER_cn.md)

- Advanced development
  - [New data augmentations](docs/advanced_tutorials/READER.md)
  - [New detection algorithms](docs/advanced_tutorials/MODEL_TECHNICAL.md)


## Model Zoo

- Universal object detection
  - [Model library and baselines](docs/MODEL_ZOO_cn.md)
  - [Mobile models](configs/mobile/README.md)
  - [Anchor free models](configs/anchor_free/README.md)
  - [PP-YOLO](configs/ppyolo/README_cn.md)
  - [676 classes of object detection](docs/featured_model/LARGE_SCALE_DET_MODEL.md)
  - [Two-stage practical PSS-Det](configs/rcnn_enhance/README.md)
- Universal instance segmentation
  - [SOLOv2](configs/solov2/README.md)
- Vertical field
  - [Face detection](docs/featured_model/FACE_DETECTION.md)
  - [Pedestrian detection](docs/featured_model/CONTRIB_cn.md)
  - [Vehicle detection](docs/featured_model/CONTRIB_cn.md)
- Competition Plan
  - [Objects365 2019 Challenge champion model](docs/featured_model/champion_model/CACascadeRCNN.md)
  - [Best single model of Open Images 2019-Object Detction](docs/featured_model/champion_model/OIDV5_BASELINE_MODEL.md)

## Applications

- [Christmas portrait automatic generation tool](application/christmas)

## Updates

v2.0-rc was released at `02/2021`, add dygraph version, which supports RCNN, YOLOv3, PP-YOLO, SSD/SSDLite, FCOS, TTFNet, SOLOv2, etc. supports model pruning and quantization, supports deploying and accelerating by TensorRT, etc. Please refer to [change log](docs/CHANGELOG.md) for details.


## License

PaddleDetection is released under the [Apache 2.0 license](LICENSE).


## Contributing

Contributions are highly welcomed and we would really appreciate your feedback!!
