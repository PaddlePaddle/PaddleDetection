English | [简体中文](README.md)

# PaddleDetection

PaddleDetection is an end-to-end object detection development kit based on PaddlePaddle, which
aims to help developers in the whole development of training models, optimizing performance and
inference speed, and deploying models. PaddleDetection provides varied object detection architectures
in modular design, and wealthy data augmentation methods, network components, loss functions, etc.
PaddleDetection supported practical projects such as industrial quality inspection, remote sensing
image object detection, and automatic inspection with its practical features such as model compression
and multi-platform deployment.

**Now all models in PaddleDetection require PaddlePaddle version 1.7 or higher, or suitable develop version.**

<div align="center">
  <img src="docs/images/000000570688.jpg" />
</div>


## Introduction

Features:

- Production Ready:

  Key operations are implemented in C++ and CUDA, together with PaddlePaddle's
highly efficient inference engine, enables easy deployment in server environments.

- Highly Flexible:

  Components are designed to be modular. Model architectures, as well as data
preprocess pipelines, can be easily customized with simple configuration
changes.

- Performance Optimized:

  With the help of the underlying PaddlePaddle framework, faster training and
reduced GPU memory footprint is achieved. Notably, YOLOv3 training is
much faster compared to other frameworks. Another example is Mask-RCNN
(ResNet50), we managed to fit up to 4 images per GPU (Tesla V100 16GB) during
multi-GPU training.

Supported Architectures:

|                     | ResNet | ResNet-vd <sup>[1](#vd)</sup> | ResNeXt-vd | SENet | MobileNet |  HRNet | Res2Net |
| ------------------- | :----: | ----------------------------: | :--------: | :---: | :-------: |:------:|:-----:  |
| Faster R-CNN        |   ✓    |                             ✓ |     x      |   ✓   |     ✗     |   ✗    |  ✗      |
| Faster R-CNN + FPN  |   ✓    |                             ✓ |     ✓      |   ✓   |     ✗     |   ✓    |  ✓      |
| Mask R-CNN          |   ✓    |                             ✓ |     x      |   ✓   |     ✗     |   ✗    |  ✗      |
| Mask R-CNN + FPN    |   ✓    |                             ✓ |     ✓      |   ✓   |     ✗     |   ✗    |  ✓      |
| Cascade Faster-RCNN |   ✓    |                             ✓ |     ✓      |   ✗   |     ✗     |   ✗    |  ✗      |
| Cascade Mask-RCNN   |   ✓    |                             ✗ |     ✗      |   ✓   |     ✗     |   ✗    |  ✗      |
| Libra R-CNN         |   ✗    |                             ✓ |     ✗      |   ✗   |     ✗     |   ✗    |  ✗      |
| RetinaNet           |   ✓    |                             ✗ |     ✗      |   ✗   |     ✗     |   ✗    |  ✗      |
| YOLOv3              |   ✓    |                             ✗ |     ✗      |   ✗   |     ✓     |   ✗    |  ✗      |
| SSD                 |   ✗    |                             ✗ |     ✗      |   ✗   |     ✓     |   ✗    |  ✗      |
| BlazeFace           |   ✗    |                             ✗ |     ✗      |   ✗   |     ✗     |   ✗    |  ✗      |
| Faceboxes           |   ✗    |                             ✗ |     ✗      |   ✗   |     ✗     |   ✗    |  ✗      |

<a name="vd">[1]</a> [ResNet-vd](https://arxiv.org/pdf/1812.01187) models offer much improved accuracy with negligible performance cost.

More Backbones:

- DarkNet
- VGG
- GCNet
- CBNet

Advanced Features:

- [x] **Synchronized Batch Norm**: currently used by YOLOv3.
- [x] **Group Norm**
- [x] **Modulated Deformable Convolution**
- [x] **Deformable PSRoI Pooling**
- [x] **Non-local and GCNet**

**NOTE:** Synchronized batch normalization can only be used on multiple GPU devices, can not be used on CPU devices or single GPU device.

## Tutorials

**News:** Documentation:[https://paddledetection.readthedocs.io](https://paddledetection.readthedocs.io)

### Get Started

- [Installation guide](docs/tutorials/INSTALL.md)
- [Quick start on small dataset](docs/tutorials/QUICK_STARTED.md)
- [Train/Evaluation/Inference](docs/tutorials/GETTING_STARTED.md)
- [FAQ](docs/tutorials/FAQ.md)

### Advanced Tutorial

- [Guide to preprocess pipeline and custom dataset](docs/advanced_tutorials/READER.md)
- [Models technical](docs/advanced_tutorials/MODEL_TECHNICAL.md)
- [Introduction to the configuration workflow](docs/advanced_tutorials/CONFIG.md)
- [IPython Notebook demo](demo/mask_rcnn_demo.ipynb)
- [Transfer learning document](docs/advanced_tutorials/TRANSFER_LEARNING.md)
- [Model compression](slim)
    - [Model compression benchmark](slim)
    - [Quantization](slim/quantization)
    - [Model pruning](slim/prune)
    - [Model distillation](slim/distillation)
    - [Neural Architecture Search](slim/nas)
- [Deployment](inference)
    - [Export model for inference](docs/advanced_tutorials/inference/EXPORT_MODEL.md)
    - [Model inference](docs/advanced_tutorials/inference/INFERENCE.md)
    - [C++ inference](inference/README.md)
    - [Inference benchmark](docs/advanced_tutorials/inference/BENCHMARK_INFER_cn.md)

## Model Zoo

- Pretrained models are available in the [PaddleDetection model zoo](docs/MODEL_ZOO.md).
- [Face detection models](configs/face_detection/README.md) BlazeFace series model with the highest precision of 91.5% on Wider-Face dataset and outstanding inference performance.
- [Pretrained models for pedestrian  and vehicle detection](contrib/README.md) Models for object detection in specific scenarios.
- [YOLOv3 enhanced model](docs/YOLOv3_ENHANCEMENT.md) Compared to MAP of 33.0% in paper, enhanced YOLOv3 reaches the MAP of 43.6% and inference speed is improved as well
- [Objects365 2019 Challenge champion model](docs/CACascadeRCNN.md) One of the best single models in Objects365 Full Track of which MAP reaches 31.7%.
- [Open Images Dataset V5 and Objects365 Dataset models](docs/OIDV5_BASELINE_MODEL.md)


## License
PaddleDetection is released under the [Apache 2.0 license](LICENSE).

## Updates
v0.2.0 was released at `02/2020`, add some models，Upgrade data processing module, Split YOLOv3's loss, fix many known bugs, etc.
Please refer to [版本更新文档](docs/CHANGELOG.md) for details.

## Contributing

Contributions are highly welcomed and we would really appreciate your feedback!!
