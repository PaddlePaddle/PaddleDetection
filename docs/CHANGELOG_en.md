English | [简体中文](./CHANGELOG.md)

# Version Update Information

## Last Version Information

### 2.6(02.15/2023)

- Featured model

  - Release rotated object detector PP-YOLOE-R：SOTA Anchor-free rotated object detection model with high accuracy and efficiency. It has a series of models, named s/m/l/x, for cloud and edge devices and avoids using special operators to be deployed friendly with TensorRT.
    - Release small object detector PP-YOLOE-SOD: End-to-end detection pipeline based on sliced images and SOTA model on VisDrone based on original images.
    - Release crowded object detector: Crowded object detection model with top accuracy on SKU dataset.

- Functions in different scenarios

  - Release real-time object detection model on edge device in PP-Human v2. The model reaches 45.7mAP and 80FPS on Jetson AGX
  - Release real-time object detection model on edge device in PP-Vehicle. The model reaches 53.5mAP and 80FPS on Jetson AGX
  - Support multi-stream deployment in PP-Human v2 and PP-Vehicle. Achieved 20FPS in 4-stream deployment on Jetson AGX
  - Support retrograde and press line detection in PP-Vehicle

- Cutting-edge algorithms

  - Release YOLOv8 and YOLOv6 3.0 in YOLO Family
  - Release object detection algorithm DINO, YOLOF
  - Rich ViTDet series including PP-YOLOE+ViT_base, Mask RCNN + ViT_base, Mask RCNN + ViT_large
  - Release MOT algorithm CenterTrack
  - Release oriented object detection algorithm FCOSR
  - Release instance segmentation algorithm QueryInst
  - Release 3D keypoint detection algorithm Metro3d
  - Release distillation algorithm FGD，LD，CWD and PP-YOLOE+ distillation with improvement of 1.1+ mAP
  - Release SSOD algorithm DenseTeacher and adapt for PP-YOLOE+
  - Release few shot finetuning algorithm, including Co-tuning and Contrastive learning

- Framework capabilities

  - New functions
    - Release Grad-CAM for heatmap visualization. Support Faster RCNN, Mask RCNN, PP-YOLOE, BlazeFace, SSD, RetinaNet.
  - Improvement and fixes
    - Support python 3.10
    - Fix EMA for no-grad parameters
    - Simplify PP-YOLOE architecture
    - Support AdamW for Paddle 2.4.1

### 2.5(08.26/2022)

- Featured model

  - PP-YOLOE+：
    - Released PP-YOLOE+ model, with a 0.7%-2.4% mAP improvement on COCO test2017. 3.75 times faster model training convergence rate and 1.73-2.3 times faster end-to-end inference speed
    - Released pre-trained models for smart agriculture, night security detection, and industrial quality inspection with 1.3%-8.1% mAP accuracy improvement
    - supports 10 high-performance training deployment capabilities, including distributed training, online quantization, and serving deployment. We also provide more than five new deployment demos, such as C++/Python Serving, TRT native inference, and ONNX Runtime
  - PP-PicoDet：
    - Release the PicoDet-NPU model to support full quantization of model deployment
    - Add PicoDet layout analysis model with 0.5% mAP accuracy improvement due to FGD distillation algorithm
  - PP-TinyPose
    - Release PP-TinyPose Plus with 9.1% end-to-end AP improvement for business data sets such as physical exercises, dance, and other scenarios
    - Covers unconventional movements such as turning to one side, lying down, jumping, high lift
    - Add stabilization module (via filter) to significantly improve the stability at key points

- Functions in different scenarios

  - PP-Human v2
    - Release PP-Human v2, which supports four industrial features: behavioral recognition case zoo for multiple solutions, human attribute recognition, human traffic detection and trajectory retention, as well as high precision multi-camera tracking
    - Upgraded  underlying algorithm capabilities: 1.5% mAP improvement in pedestrian detection accuracy; 10.2% MOTA improvement in pedestrian tracking accuracy, 34% speed improvement in the lightweight model; 0.6% ma improvement in attribute recognition accuracy, 62.5% speed improvement in the lightweight model
    - Provides comprehensive tutorials covering data collection and annotation, model training optimization and prediction deployment, and post-processing code modification in the pipeline
    - Supports online video streaming input
    - Become more user-friendly with a one-line code execution function that automates the process determination and model download
  - PP-Vehicle
    - Launch PP-Vehicle, which supports four core functions for traffic application: license plate recognition, attribute recognition, traffic flow statistics, and violation detection
    - License plate recognition supports a lightweight model based on PP-OCR v3
    - Vehicle attribute recognition supports a multi-label classification model based on PP-LCNet
    - Compatible with various data input formats such as pictures, videos and online video streaming
    - Become more user-friendly with a one-line code execution function that automates the process determination and model download

- Cutting-edge algorithms

  - YOLO Family
    - Release the full range of YOLO family models covering the cutting-edge detection algorithms YOLOv5, YOLOv6 and YOLOv7
    - Based on the ConvNext backbone network, YOLO's algorithm training periods are reduced by 5-8 times with accuracy generally improving by 1%-5% mAP; Thanks to the model compression strategy, its speed increased by over 30% with no loss of precision.
  - Newly add high precision detection model based on [ViT](configs/vitdet) backbone network, with a 55.7% mAP accuracy on the COCO dataset
  - Newly add multi-object tracking model [OC-SORT](configs/mot/ocsort)
  - Newly add [ConvNeXt](configs/convnext) backbone network.

- Industrial application

  - Intelligent physical exercise recognition based on PP-TinyPose Plus
  - Fighting recognition based on PP-Human
  - Business hall visitor analysis based on PP-Human
  - Vehicle structuring analysis based on PP-Vehicle
  - PCB board defect detection based on PP-YOLOE+

- Framework capabilities

  - New functions
    - Release auto-compression tools and demos, 0.3% mAP accuracy loss for PP-YOLOE l version, while 13% speed increase for V100
    - Release PaddleServing python/C++ and ONNXRuntime deployment demos
    - Release PP-YOLOE end-to-end TensorRT deployment demo
    - Release FGC distillation algorithm with RetinaNet accuracy improved by 3.3%
    - Release distributed training documentation
  - Improvement and fixes
    - Fix compilation problem with Windows c++ deployment
    - Fix problems when saving results of inference data in VOC format
    - Fix the detection box output of FairMOT c++ deployment
    - Rotating frame detection model S2ANet supports batch size>1 deployment

### 2.4(03.24/2022)

- PP-YOLOE：
  - Release PP-YOLOE object detection models, achieve mAP as 51.6% on COCO test dataset and 78.1 FPS on Nvidia V100 by PP-YOLOE-l, reach SOTA performance for object detection on GPU``
  - Release series models: s/m/l/x, and support deployment base on TensorRT & ONNX
  - Spport AMP training and training speed is 33% faster than PP-YOLOv2

- PP-PicoDet:
  - Release enhanced models of PP-PicoDet, mAP promoted ~2% on COCO and inference speed accelerated 63% on CPU
  - Release PP-PicoDet-XS model with 0.7M parameters
  - Post-processing integrated into the network to optimize deployment pipeline

- PP-Human：
  - Release PP-Human human analysis pipeline，including pedestrian detection, attribute recognition, human tracking, multi-camera tracking, human statistics, action recognition. Supporting deployment with TensorRT
  - Release StrongBaseline model for attribute recognition
  - Release Centroid model for ReID
  - Release ST-GCN model for falldown action recognition

- Model richness:
  - Publish YOLOX object detection model, release series models: nano/tiny/s/m/l/x, and YOLOX-x achieves mAP as 51.8% on COCO val2017 dataset

- Function Optimize：
  - Optimize 20% training speed when training with EMA, improve saving method of EMA weights
  - Support saving inference results in COCO format

- Deployment Optimize：
  - Support export ONNX model by Paddle2ONNX for all RCNN models
  - Supoort export model with fused decode OP for SSD models to enhance inference speed in edge side
  - Support export NMS to TensorRT model, optmize inference speed on TensorRT

### 2.3(11.03/2021)

- Feature models:
    - Object detection: The lightweight object detection model PP-PicoDet, performace and inference speed reaches SOTA on mobile side
    - Keypoint detection: The lightweight keypoint detection model PP-TinyPose for mobile side

- Model richness:
    - Object detection:
        - Publish Swin-Transformer object detection model
        - Publish TOOD(Task-aligned One-stage Object Detection) model
        - Publish GFL(Generalized Focal Loss) object detection model
        - Publish Sniper optimization method for tiny object detection, supporting Faster RCNN and PP-YOLO series models
        - Publish PP-YOLO optimized model PP-YOLO-EB for EdgeBoard
    - Multi-object tracking:
        - Publish Real-time tracking system PP-Tracking
        - Publish high-precision, small-scale and lightweight model based on FairMot
        - Publish real-time tracking model zoo for pedestrian, head and vehicle tracking, including scenarios such as aerial surveillance, autonomous driving, dense crowds, and tiny object tracking
        - DeepSort support PP-YOLO, PP-PicoDet as object detector
    - Keypoint detection:
        - Publish Lite HRNet model

- Inference deployment:
    - Support NPU deployment for YOLOv3 series
    - Support C++ deployment for FairMot
    - Support C++ and PaddleLite deployment for keypoint detection series model

- Documents:
    - Add series English documents


### 2.2(08.10/2021)

- Model richness:
    - Publish the Transformer test model: DETR, Deformable DETR, Sparse RCNN
    - Key point test new Dark model, release Dark HRNet model
    - Publish the MPII dataset HRNet keypoint detection model
    - Release head and vehicle tracking vertical model

- Model optimization:
    - AlignConv optimization model was released by S2ANet, and DOTA dataset mAP was optimized to 74.0

- Inference deployment
    - Mainstream models support batch size>1 predictive deployment, including YOLOv3, PP-YOLO, Faster RCNN, SSD, TTFNet,  FCOS
    - New addition of target tracking models (JDE, Fair Mot, Deep Sort) Python side prediction deployment support, and support for TensorRT prediction
    - FairMot joint key point detection model deployment Python side predictive deployment support
    - Added support for key point detection model combined with PP-YOLO prediction deployment

- Documents:
    - New TensorRT version notes to Windows Predictive Deployment documentation
    - FAQ documents are updated

- Bug fixes:
    - Fixed PP-YOLO series model training convergence problem
    - Fixed the problem of no label data training when batch_size > 1


### 2.1(05.20/2021)
- Model richness enhancement:
    - Key point model: HRNet, HigherHRNet
    - Publish the multi-target tracking model: DeepSort, FairMot, JDE

- Basic framework Capabilities:
    - Supports training without labels

- Forecast deployment:
    - Paddle Inference YOLOv3 series model support batch_size>1 prediction
    - Rotating frame detection S2ANet model prediction deployment is open
    - Incremental quantization model benchmark
    - Add dynamic graph model and static graph model: Paddle-Lite demo

- Detection model compression:
    - Release PP-YOLO series model compression model

- Documents:
    - Update quick start, forecast deployment and other tutorial documentation
    - Added ONNX model export tutorial
    - Added the mobile deployment document


### 2.0(04.15/2021)

  **Description:** Since version 2.0, dynamic graphs are used as the default version of Paddle Detection, the original `dygraph` directory is switched to the root directory, and the original static graph implementation is moved to the `static` directory.

  - Enhancement of dynamic graph model richness:
    - PP-YOLOv2 and PP-YOLO tiny models were published. The accuracy of PP-YOLOv2 COCO Test dataset reached 49.5%, and the prediction speed of V100 reached 68.9 FPS
    - Release the rotary frame detection model S2ANet
    - Release the two-phase utility model PSS-Det
    - Publish the face detection model Blazeface

  - New basic module:
    - Added SENet, GhostNet, and Res2Net backbone networks
    - Added VisualDL training visualization support
    - Added single precision calculation and PR curve drawing function
    - The YOLO models support THE NHWC data format

  - Forecast deployment:
    - Publish forecast benchmark data for major models
    - Adaptive to TensorRT6, support TensorRT dynamic size input, support TensorRT int8 quantitative prediction
    - 7 types of models including PP-YOLO, YOLOv3, SSD, TTFNet, FCOS, Faster RCNN are deployed in Python/CPP/TRT prediction on Linux, Windows and NV Jetson platforms

  - Detection model compression:
    - Distillation: Added dynamic map distillation support and released YOLOv3-MobileNetV1 distillation model
    - Joint strategy: new dynamic graph prunning + distillation joint strategy compression scheme, and release YOLOv3-MobileNetV1 prunning + distillation compression model
    - Problem fix: Fixed dynamic graph quantization model export problem

  - Documents:
    - New English document of dynamic graph: including homepage document, getting started, quick start, model algorithm, new dataset, etc
    - Added both English and Chinese installation documents of dynamic diagrams
    - Added configuration file templates and description documents of dynamic graph RCNN series and YOLO series


## Historical Version Information

### 2.0-rc(02.23/2021)
  - Enhancement of dynamic graph model richness:
    - Optimize networking and training mode of RCNN models, and improve accuracy of RCNN series models (depending on Paddle Develop or version 2.0.1)
    - Added support for SSDLite, FCOS, TTFNet, SOLOv2 series models
    - Added pedestrian and vehicle vertical object detection models

  - New dynamic graph basic module:
    - Added MobileNetV3 and HRNet backbone networks
    - Improved roi-align calculation logic for RCNN series models (depending on Paddle Develop or version 2.0.1)
    - Added support for Synchronized Batch Norm
    - Added support for Modulated Deformable Convolution

  - Forecast deployment:
    - Publish dynamic diagrams in python, C++, and Serving deployment solution and documentation. Support Faster RCNN, Mask RCNN, YOLOv3, PPYOLO, SSD, TTFNet, FCOS, SOLOv2 and other models to predict deployment
    - Dynamic graph prediction deployment supports TensorRT mode FP32, FP16 inference acceleration

  - Detection model compression:
    - Prunning: Added dynamic graph prunning support, and released YOLOv3-MobileNetV1 prunning model
    - Quantization: Added quantization support of dynamic graph, and released quantization models of YOLOv3-MobileNetV1 and YOLOv3-MobileNetV3

  - Documents:
    - New Dynamic Diagram tutorial documentation: includes installation instructions, quick start, data preparation, and training/evaluation/prediction process documentation
    - New advanced tutorial documentation for dynamic diagrams: includes documentation for model compression and inference deployment
    - Added dynamic graph model library documentation

### v2.0-beta(12.20/2020)
  - Dynamic graph support:
    -  Support for Faster-RCNN, Mask-RCNN, FPN, Cascade Faster/Mask RCNN, YOLOv3 and SSD models, trial version.
  - Model upgrade:
    - Updated PP-YOLO Mobile-Netv3 large and small models with improved accuracy, and added prunning and distillation models.
  - New features:
    - Support VisualDL visual data preprocessing pictures.

  - Bug fix:
    - Fix Blaze Face keypoint prediction bug.


### v0.5.0(11/2020)
  - Model richness enhancement:
    - SOLOv2 series models were released, in which the SOLOv2-Light-R50-VD-DCN-FPN model achieved 38.6 FPS on a single gpu V100, accelerating by 24%, and the accuracy of COCO verification set reached 38.8%, improving by 2.4 absolute percentage points.
    - Added Android mobile terminal detection demo, including SSD, YOLO series model, can directly scan code installation experience.

  - Mobile terminal model optimization:
    - Added to PACT's new quantization strategy, YOLOv3 Mobilenetv3 is 0.7% better than normal quantization on COCO datasets.

  - Ease of use and functional components:
    - Enhance the function of generate_proposal_labels operator to avoid nan risk of the model.
    - Fixed several problems with deploy python and C++ prediction.
    - Unified COCO and VOC datasets under the evaluation process, support the output of a single class of AP and P-R curves.
    - PP-YOLO supports rectangular input images.

  - Documents:
    - Added object detection whole process tutorial, added Jetson platform deployment tutorial.


### v0.4.0(07/2020)
  - Model richness enhancement:
    - The PPYOLO model was released. The accuracy of COCO dataset reached 45.2%, and the prediction speed of single gpu V100 reached 72.9 FPS, which was better than that of YOL Ov4 model.
    - New TTFNet model, base version aligned with competing products, COCO dataset accuracy up to 32.9%.
    - New HTC model, base version aligned with competing products, COCO dataset accuracy up to 42.2%.
    - BlazeFace key point detection model was added, with an accuracy of 85.2% in Wider-Face's Easy-Set.
    - ACFPN model was added, and the accuracy of COCO dataset reached 39.6%.
    - General object detection model (including 676 classes) on the publisher side. On the COCO dataset with the same strategy, when V100 is 19.5FPS, the COCO mAP can reach 49.4%.

  - Mobile terminal model optimization:
    - Added SSD Lite series optimization models, including Ghost Net Backbone, FPN components, etc., with accuracy improved by 0.5% and 1.5%.

  - Ease of use and functional components:
    - Add GridMask, Random Erasing data enhancement method.
    - Added support for Matrix NMS.
    - EMA(Exponential Moving Average) training support.
    - The new multi-machine training method, the average acceleration ratio of two machines to single machine is 80%, multi-machine training support needs to be further verified.

### v0.3.0(05/2020)
  - Model richness enhancement:
    - Efficientdet-D0 model added, speed and accuracy is better than competing products.
    - Added YOLOv4 prediction model, precision aligned with competing products; Added YOLOv4 fine tuning training on Pascal VOC datasets with accuracy of 85.5%.
    - YOLOv3 added MobileNetV3 backbone network, COCO dataset accuracy reached 31.6%.
    - Add Anchor-free model FCOS, the accuracy is better than competing products.
    - Anchor-free model Cornernet Squeeze was added, the accuracy was better than competing products, and the accuracy of COCO dataset of optimized model was 38.2% and +3.7%, 5% faster than YOL Ov3 Darknet53.
    - The CascadeRCNN-ResNet50vd model, which is a practical object detection model on the server side, is added, and its speed and accuracy are better than that of the competitive EfficientDet.

  - Mobile terminal launched three models:
    - SSSDLite model: SSDLite-Mobilenetv3 small/large model, with better accuracy than competitors.
    - YOLOv3 Mobile solution: The YOLOv3-MobileNetv3 model accelerates 3.5 times after compression, which is faster and more accurate than the SSD Lite model of competing products.
    - RCNN Mobile terminal scheme: CascadeRCNN-MobileNetv3, after series optimization, launched models with input images of 320x320 and 640x640 respectively, with high cost performance for speed and accuracy.

  - Anticipate deployment refactoring:
    - New Python prediction deployment process, support for RCNN, YOLO, SSD, Retina Net, face models, support for video prediction.
    - Refactoring C++ predictive deployment to improve ease of use.

  - Ease of use and functional components:
    - Added Auto Augment data enhancement.
    - Upgrade the detection library document structure.
    - Support shape matching automatically by transfer learning.
    - Optimize memory footprint during mask branch evaluation.

### v0.2.0(02/2020)
  - The new model:
    - Added CBResNet model.
    - Added LibraRCNN model.
    - The accuracy of YOLOv3 model was further improved, and the accuracy based on COCO data reached 43.2%, 1.4% higher than the previous version.
  - New Basic module:
    - Trunk network: CBResNet is added.
    - Loss module: Loss of YOLOv3 supports fine-grained OP combinations.
    - Regular module: Added the Drop Block module.
  - Function optimization and improvement:
    - Accelerate YOLOv3 data preprocessing and increase the overall training speed by 40%.
    - Optimize data preprocessing logic to improve ease of use.
    - dd face detection prediction benchmark data.
    - Added C++ prediction engine Python API prediction example.
  - Detection model compression:
    - prunning: Release MobileNet-YOLOv3 prunning scheme and model, based on VOC data FLOPs 69.6%, mAP + 1.4%, based on COCO DATA FLOPS 28.8%, mAP + 0.9%; Release ResNet50vd-DCN-YOLOv3 clipped solution and model based on COCO datasets 18.4%, mAP + 0.8%.
    - Distillation: Release MobileNet-YOLOv3 distillation scheme and model, based on VOC data mAP + 2.8%, COCO data mAP + 2.1%.
    - Quantification: Release quantification models of YOLOv3 Mobile Net and Blaze Face.
    - Prunning + distillation: release MobileNet-YOLOv3 prunning + distillation solution and model, 69.6% based on COCO DATA FLOPS, 64.5% based on TensorRT prediction acceleration, 0.3% mAP; Release ResNet50vd-DCN-YOLOv3 tailoring + distillation solution and model, 43.7% based on COCO Data FLOPS, 24.0% based on TensorRT prediction acceleration, mAP + 0.6%.
    - Search: Open source Blaze Face Nas complete search solution.
  - Predict deployment:
    - Integrated TensorRT, support FP16, FP32, INT8 quantitative inference acceleration.
  - Document:
    - Add detailed data preprocessing module to introduce documents and implement custom data Reader documents.
    - Added documentation on how to add algorithm models.
    - Document deployment to the web site: https://paddledetection.readthedocs.io

### 12/2019
- Add Res2Net model.
- Add HRNet model.
- Add GIOU loss and DIOU loss。


### 21/11/2019
- Add CascadeClsAware RCNN model.
- Add CBNet, ResNet200 and Non-local model.
- Add SoftNMS.
- Add Open Image V5 dataset and Objects365 dataset model

### 10/2019
- Added enhanced YOLOv3 model with accuracy up to 41.4%.
- Added Face detection models BlazeFace and Faceboxes.
- Rich COCO based models, accuracy up to 51.9%.
- Added CA-Cascade-RCNN, one of the best single models to win on Objects365 2019 Challenge.
- Add pedestrian detection and vehicle detection pre-training models.
- Support FP16 training.
- Added cross-platform C++ inference deployment scheme.
- Add model compression examples.


### 2/9/2019
- Add GroupNorm model.
- Add CascadeRCNN+Mask model.

### 5/8/2019
- Add Modulated Deformable Convolution series model

### 29/7/2019

- Add detection library Chinese document
- Fixed an issue where R-CNN series model training was evaluated simultaneously
- Add ResNext101-vd + Mask R-CNN + FPN models
- Added YOLOv3 model based on VOC dataset

### 3/7/2019

- First release of PaddleDetection Detection library and Detection model library
- models：Faster R-CNN, Mask R-CNN, Faster R-CNN+FPN, Mask
  R-CNN+FPN, Cascade-Faster-RCNN+FPN, RetinaNet, YOLOv3, 和SSD.
