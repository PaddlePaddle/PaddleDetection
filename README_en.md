English | [简体中文](README_cn.md)

<div align="center">
<p align="center">
  <img src="https://user-images.githubusercontent.com/48054808/160532560-34cf7a1f-d950-435e-90d2-4b0a679e5119.png" align="middle" width = "800" />
</p>

****A High-Efficient Development Toolkit for Object Detection based on [PaddlePaddle](https://github.com/paddlepaddle/paddle).****

<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href="https://github.com/PaddlePaddle/PaddleDetection/releases"><img src="https://img.shields.io/github/v/release/PaddlePaddle/PaddleDetection?color=ffa"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href="https://github.com/PaddlePaddle/PaddleDetection/stargazers"><img src="https://img.shields.io/github/stars/PaddlePaddle/PaddleDetection?color=ccf"></a>

</div>

<div  align="center">
  <img src="docs/images/ppdet.gif" width="800"/>

</div>

## <img src="https://user-images.githubusercontent.com/48054808/157793354-6e7f381a-0aa6-4bb7-845c-9acf2ecc05c3.png" width="20"/> Latest News

- 🔥 **2022.8.09：Release [YOLO series model zoo](https://github.com/nemonameless/PaddleDetection_YOLOSeries)**
  - Comprehensive coverage of classic and latest models of the YOLO series: Including YOLOv3，Paddle real-time object detection model PP-YOLOE, and frontier detection algorithms YOLOv4, YOLOv5, YOLOX, MT-YOLOv6 and YOLOv7
  - Better model performance：Upgrade based on various YOLO algorithms, shorten training time in 5-8 times and the accuracy is generally improved by 1%-5% mAP. The model compression strategy is used to achieve 30% improvement in speed without precision loss
  - Complete end-to-end development support：End-to-end development pipieline including training, evaluation, inference, model compression and deployment on various hardware. Meanwhile, support flexible algorithnm switch and implement customized development efficiently

- 🔥 **2022.8.01：Release [PP-TinyPose plus](./configs/keypoint/tiny_pose/). The end-to-end precision improves 9.1% AP in dataset
 of fitness and dance scenes**
  - Increase data of sports scenes, and the recognition performance of complex actions is significantly improved, covering actions such as sideways, lying down, jumping, and raising legs
  - Detection model uses PP-PicoDet plus and the precision on COCO dataset is improved by 3.1% mAP
  - The stability of keypoints is enhanced. Implement the filter stabilization method to make the video prediction result more stable and smooth.

- 2022.7.14：Release [pedestrian analysis tool PP-Human v2](./deploy/pipeline)
  - Four major functions: five complicated action recognition with high performance and Flexible, real-time human attribute recognition, visitor flow statistics and high-accuracy multi-camera tracking.
  - High performance algorithm: including pedestrian detection, tracking, attribute recognition which is robust to the number of targets and the variant of background and light.
  - Highly Flexible: providing complete introduction of end-to-end development and optimization strategy, simple command for deployment and compatibility with different input format.

- 2022.3.24：PaddleDetection released[release/2.4 version](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4)  
  - Release high-performanace SOTA object detection model [PP-YOLOE](configs/ppyoloe). It integrates cloud and edge devices and provides S/M/L/X versions. In particular, Verson L has the accuracy as 51.4% on COCO test 2017 dataset, inference speed as 78.1 FPS on a single Test V100. It supports mixed precision training, 33% faster than PP-YOLOv2. Its full range of multi-sized models can meet different hardware arithmetic requirements, and adaptable to server, edge-device GPU and other AI accelerator cards on servers.
  - Release ultra-lightweight SOTA object detection model [PP-PicoDet Plus](configs/picodet) with 2% improvement in accuracy and 63% improvement in CPU inference speed. Add PicoDet-XS model with a 0.7M parameter, providing model sparsification and quantization functions for model acceleration. No specific post processing module is required for all the hardware, simplifying the deployment.  
  - Release the real-time pedestrian analysis tool [PP-Human](deploy/pphuman). It has four major functions: pedestrian tracking, visitor flow statistics, human attribute recognition and falling detection. For falling detection, it is optimized based on real-life data with accurate recognition of various types of falling posture. It can adapt to different environmental background, light and camera angle.
  - Add [YOLOX](configs/yolox) object detection model with nano/tiny/S/M/L/X. X version has the accuracy as 51.8% on COCO  Val2017 dataset.

- [More releases](https://github.com/PaddlePaddle/PaddleDetection/releases)

## <img title="" src="https://user-images.githubusercontent.com/48054808/157795569-9fc77c85-732f-4870-9be0-99a7fe2cff27.png" alt="" width="20"> Introduction

PaddleDetection is an end-to-end object detection development kit based on PaddlePaddle, which implements varied mainstream object detection, instance segmentation, tracking and keypoint detection algorithms in modular design with configurable modules such as network components, data augmentations and losses. It releases many kinds SOTA industry practice models and integrates abilities of model compression and cross-platform high-performance deployment to help developers in the whole process with a faster and better way.

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
  <img src="https://user-images.githubusercontent.com/22989727/183843004-baebf75f-af7c-4a7c-8130-1497b9a3ec7e.png"  width = "200" />  
  <img src="https://user-images.githubusercontent.com/34162360/177678712-4655747d-4290-4ad9-b7a1-4564a5418ac6.jpg"  width = "200" />  
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
            <li>PP-YOLOE</li>
            <li>YOLOX</li>
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
            <li>DeepSORT</li>
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
          <li>AugmentHSV</li>
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

- `PP-YOLOE` is optimized version of `PP-YOLO v2` which has mAP of 51.6% and 78.1FPS on Tesla V100

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

- [Installation Guide](docs/tutorials/INSTALL.md)
- [Prepare Dataset](docs/tutorials/PrepareDataSet_en.md)
- [Quick Start on PaddleDetection](docs/tutorials/GETTING_STARTED.md)

### Advanced Tutorials

- Parameter Configuration

  - [Parameter configuration for RCNN model](docs/tutorials/config_annotation/faster_rcnn_r50_fpn_1x_coco_annotation_en.md)
  - [Parameter configuration for PP-YOLO model](docs/tutorials/config_annotation/ppyolo_r50vd_dcn_1x_coco_annotation_en.md)

- Model Compression(Based on [PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim))

  - [Prune/Quant/Distill](configs/slim)

- Inference and Deployment

  - [Export model for inference](deploy/EXPORT_MODEL_en.md)
  - [Paddle Inference](deploy/README_en.md)
    - [Python inference](deploy/python)
    - [C++ inference](deploy/cpp)
  - [Paddle-Lite](deploy/lite)
  - [Paddle Serving](deploy/serving)
  - [Export ONNX model](deploy/EXPORT_ONNX_MODEL_en.md)
  - [Inference benchmark](deploy/BENCHMARK_INFER_en.md)
  - [Exporting to ONNX and using OpenVINO for inference](docs/advanced_tutorials/openvino_inference/README.md)

- Advanced Development

  - [New data augmentations](docs/advanced_tutorials/READER_en.md)
  - [New detection algorithms](docs/advanced_tutorials/MODEL_TECHNICAL.md)

## <img src="https://user-images.githubusercontent.com/48054808/157829890-a535b8a6-631c-4c87-b861-64d4b32b2d6a.png" width="20"/> Model Zoo

- General Object Detection
  - [Model library and baselines](docs/MODEL_ZOO_cn.md)
  - [PP-YOLOE](configs/ppyoloe/README_cn.md)
  - [PP-YOLO](configs/ppyolo/README.md)
  - [PP-PicoDet](configs/picodet/README.md)
  - [Enhanced Anchor Free model--TTFNet](configs/ttfnet/README_en.md)
  - [Mobile models](static/configs/mobile/README_en.md)
  - [676 classes of object detection](static/docs/featured_model/LARGE_SCALE_DET_MODEL_en.md)
  - [Two-stage practical PSS-Det](configs/rcnn_enhance/README_en.md)
  - [SSLD pretrained models](docs/feature_models/SSLD_PRETRAINED_MODEL_en.md)
- General Instance Segmentation
  - [SOLOv2](configs/solov2/README.md)
- Rotated Object Detection
  - [S2ANet](configs/dota/README_en.md)
- [Keypoint Detection](configs/keypoint)
  - [PP-TinyPose](configs/keypoint/tiny_pose)
  - HigherHRNet
  - HRNet
  - LiteHRNet
- [Multi-Object Tracking](configs/mot/README.md)
  - [PP-Tracking](deploy/pptracking/README_en.md)
  - [DeepSORT](configs/mot/deepsort/README.md)
  - [JDE](configs/mot/jde/README.md)
  - [FairMOT](configs/mot/fairmot/README.md)
  - [ByteTrack](configs/mot/bytetrack/README.md)
- Practical Specific Models
  - [Face detection](configs/face_detection/README_en.md)
  - [Pedestrian detection](configs/pedestrian/README.md)
  - [Vehicle detection](configs/vehicle/README.md)
- Scienario Solution
  - [Real-Time Human Analysis Tool PP-Human](deploy/pphuman)
- Competition Solution
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
- Thanks [manangoel99](https://github.com/manangoel99) for contributing Wandblogger for visualization of the training and evaluation metrics  

## <img src="https://user-images.githubusercontent.com/48054808/157835276-9aab9d1c-1c46-446b-bdd4-5ab75c5cfa48.png" width="20"/> Citation

```
@misc{ppdet2019,
title={PaddleDetection, Object detection and instance segmentation toolkit based on PaddlePaddle.},
author={PaddlePaddle Authors},
howpublished = {\url{https://github.com/PaddlePaddle/PaddleDetection}},
year={2019}
}
```
