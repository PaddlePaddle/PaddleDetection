English | [简体中文](README_cn.md)

# PaddleDetection

The goal of PaddleDetection is to provide easy access to a wide range of object
detection models in both industry and research settings. We design
PaddleDetection to be not only performant, production-ready but also highly
flexible, catering to research needs.


<div align="center">
  <img src="demo/output/000000570688.jpg" />
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
reduced GPU memory footprint is achieved. Notably, Yolo V3 training is
much faster compared to other frameworks. Another example is Mask-RCNN
(ResNet50), we managed to fit up to 4 images per GPU (Tesla V100 16GB) during
multi-GPU training.

Supported Architectures:

|                    | ResNet | ResNet-vd <sup>[1](#vd)</sup> | ResNeXt-vd | SENet | MobileNet | DarkNet | VGG |
|--------------------|:------:|------------------------------:|:----------:|:-----:|:---------:|:-------:|:---:|
| Faster R-CNN       | ✓      |                             ✓ | x          | ✓     | ✗         | ✗       | ✗   |
| Faster R-CNN + FPN | ✓      |                             ✓ | ✓          | ✓     | ✗         | ✗       | ✗   |
| Mask R-CNN         | ✓      |                             ✓ | x          | ✓     | ✗         | ✗       | ✗   |
| Mask R-CNN + FPN   | ✓      |                             ✓ | ✓          | ✓     | ✗         | ✗       | ✗   |
| Cascade R-CNN      | ✓      |                             ✗ | ✗          | ✗     | ✗         | ✗       | ✗   |
| RetinaNet          | ✓      |                             ✗ | ✗          | ✗     | ✗         | ✗       | ✗   |
| Yolov3             | ✓      |                             ✗ | ✗          | ✗     | ✓         | ✓       | ✗   |
| SSD                | ✗      |                             ✗ | ✗          | ✗     | ✓         | ✗       | ✓   |

<a name="vd">[1]</a> [ResNet-vd](https://arxiv.org/pdf/1812.01187) models offer much improved accuracy with negligible performance cost.

Advanced Features:

- [x] **Synchronized Batch Norm**: currently used by Yolo V3.
- [x] **Group Norm**: pretrained models to be released.
- [x] **Modulated Deformable Convolution**: pretrained models to be released.
- [x] **Deformable PSRoI Pooling**: pretrained models to be released.

**NOTE:** Synchronized batch normalization can only be used on multiple GPU devices, can not be used on CPU devices or single GPU device.


## Model zoo

Pretrained models are available in the PaddlePaddle [PaddleDetection model zoo](docs/MODEL_ZOO.md).


## Installation

Please follow the [installation guide](docs/INSTALL.md).


## Get Started

For inference, simply run the following command and the visualized result will
be saved in `output`.

```bash
export PYTHONPATH=`pwd`:$PYTHONPATH
python tools/infer.py -c configs/mask_rcnn_r50_1x.yml \
    -o weights=https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r50_1x.tar \
    --infer_img=demo/000000570688.jpg
```

For detailed training and evaluation workflow, please refer to [GETTING_STARTED.md](docs/GETTING_STARTED.md).

For detailed configuration and parameter description, please refer to [Complete config files](docs/config_example/)

We also recommend users to take a look at the [IPython Notebook demo](demo/mask_rcnn_demo.ipynb)

Further information can be found in these documentations:

- [Introduction to the configuration workflow.](docs/CONFIG.md)
- [Guide to custom dataset and preprocess pipeline.](docs/DATA.md)


##  Todo List

Please note this is a work in progress, substantial changes may come in the
near future.
Some of the planned features include:

- [ ] Mixed precision training.
- [ ] Distributed training.
- [ ] Inference in 8-bit mode.
- [ ] User defined operations.
- [ ] Larger model zoo.


## Updates

#### 7/29/2019

- Update Chinese docs for PaddleDetection
- Fix bug in R-CNN models when train and test at the same time
- Add ResNext101-vd + Mask R-CNN + FPN models
- Add Yolo v3 on VOC models

#### 7/3/2019

- Initial release of PaddleDetection and detection model zoo
- Models included: Faster R-CNN, Mask R-CNN, Faster R-CNN+FPN, Mask
  R-CNN+FPN, Cascade-Faster-RCNN+FPN, RetinaNet, Yolo v3, and SSD.


## Contributing

Contributions are highly welcomed and we would really appreciate your feedback!!
