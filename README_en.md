简体中文 | [English](README_en.md)

<div align="center">
<p align="center">
  <img src="https://user-images.githubusercontent.com/48054808/160532560-34cf7a1f-d950-435e-90d2-4b0a679e5119.png" align="middle" width = "800" />
</p>

**A High-Efficient Development Toolkit for Object Detection based on [PaddlePaddle](https://github.com/paddlepaddle/paddle)**

<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href="https://github.com/PaddlePaddle/PaddleDetection/releases"><img src="https://img.shields.io/github/v/release/PaddlePaddle/PaddleDetection?color=ffa"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href="https://github.com/PaddlePaddle/PaddleDetection/stargazers"><img src="https://img.shields.io/github/stars/PaddlePaddle/PaddleDetection?color=ccf"></a>
</p>
</div>

<div  align="center">
  <img src="docs/images/ppdet.gif" width="800"/>

</div>

## <img src="https://user-images.githubusercontent.com/48054808/157793354-6e7f381a-0aa6-4bb7-845c-9acf2ecc05c3.png" width="20"/> Product Update

- 🔥 **2022.3.24：PaddleDetection released[release/2.4 version](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4)**
  
  - Release high-performanace SOTA object detection model [PP-YOLOE](configs/ppyoloe). It integrates cloud and edge devices and provides s/m/l/x versions. In particular, Verson I has the accuracy as 51.4% on COCO test 2017 dataset, inference speed as 78.1 FPS on a single Test V100. It supports mixed precision training, 33% faster than PP-YOLOv2. Its full range of multi-sized models can meet different hardware arithmetic requirements, and adaptable to server, edge-device GPU and other AI accelerator cards on servers.
  - Release ultra-lightweight SOTA object detection model [PP-PicoDet Plus](configs/picodet) with 2% improvement in accuracy and 63% improvement in CPU inference speed. Add PicoDet-XS model with a 0.7M parameter, providing model sparsification and quantization functions for model acceleration. No specific post processing module is required for all the hardware, simplifying the deployment.  
  - Release the real-time pedestrian analysis tool [PP-Human](deploy/pphuman). It has four major functions: pedestrian tracking, visitor flow statistics, human attribute recognition and falling detection. For falling detection, it is optimized based on real-life data with accurate recognition of various types of falling posture. It can adapt to different environmental background, light and camera angle.
  - Add [YOLOX](configs/yolox) object detection model with nano/tiny/s/m/l/x. X version has the accuracy as 51.8% on COCO  Val2017 dataset.

- 2021.11.03: PaddleDetection released [release/2.3 version](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3)
  
  - Release light-weight featured detection model ⚡[PP-PicoDet](configs/picodet). With a 0.99m parameter, its inference speed could reach to 150FPS when COCO mAP as over 30% 
  - Release light-weight keypoint special model ⚡[PP-TinyPose](configs/keypoint/tiny_pose), FP16 inference speed as 122 FPS  and on a single person detection. It has high performance and fast speed, unlimited detection headcounts while being effective on small objects.
  - Release real-time tracking system [PP-Tracking](deploy/pptracking), covering pedestrian, vehicle and multi-category tracking with single and multi-camera, optimization for small and intensive objects, providing technical solutions for human and vehicle traffic.
  - Add object detection models [Swin Transformer](configs/faster_rcnn)，[TOOD](configs/tood)，[GFL](configs/gfl)
  - Release optimized small object detection model [Sniper](configs/sniper) and [PP-YOLO-EB](configs/ppyolo) model which optimized for EdgeBoard
  - Add light-weight keypoint model [Lite HRNet](configs/keypoint) and supported Paddle Lite deployment

- [More releases](https://github.com/PaddlePaddle/PaddleDetection/releases)

## <img title="" src="https://user-images.githubusercontent.com/48054808/157795569-9fc77c85-732f-4870-9be0-99a7fe2cff27.png" alt="" width="20"> Brief Introduction

**PaddleDetection** is an end-to-end object detection development kit based on PaddlePaddle. Providing **over 30 model algorithm** and **over 250 pre-trained models**, it covers object detection, instance segmentation, keypoint detection, multi-object tracking. In particular, PaddleDetection offers **high- performance & light-weight** industrial SOTA models on **servers and mobile** devices, champion solution and cutting-edge algorithm. PaddleDetection provides various data augmentation methods, configurable network components, loss functions and other advanced optimization & deployment schemes. In addition to running through the whole process of data processing, model development, training, compression and deployment, PaddleDetection also provides rich cases and tutorials to accelerate the industrial application of algorithm.

<div  align="center">
  <img src="https://user-images.githubusercontent.com/48054808/157826886-2e101a71-25a2-42f5-bf5e-30a97be28f46.gif" width="800"/>
</div>

## <img src="https://user-images.githubusercontent.com/48054808/157799599-e6a66855-bac6-4e75-b9c0-96e13cb9612f.png" width="20"/> Features

- **Rich model library**: PaddleDetection provides over 250 pre-trained models including **object detection, instance segmentation, face recognition, multi-object tracking**. It covers a variety of **global competition champion** schemes.
- **Simple to use**: Modular design, decoupling each network component, easy for developers to build and try various detection models and optimization strategies, quick access to high-performance, customized algorithm.
- **Getting Through End to End**: PaddlePaddle gets through end to end from data augmentation, constructing models, training, compression, depolyment. It also supports multi-architecture, multi-device deployment for **cloud and edge** device.
- **High Performance**: Due to the high performance core, PaddlePaddle has clear advantages in training speed and memory occupation. It also supports FP16 training and multi-machine training.

<div  align="center">
  <img src="https://pcsdata.baidu.com/thumbnail/d775c7c64v0f6dea0348c33be01fb266?fid=2573618645-16051585-48700465526764&rt=pr&sign=FDTAER-yUdy3dSFZ0SVxtzShv1zcMqd-CDZxwEyVXlxu4BbVu4vukRc%2BWOQ%3D&expires=2h&chkv=0&chkbd=0&chkpc=&dp-logid=8893245804304207454&dp-callid=0&time=1656914400&bus_no=26&size=c1600_u1600&quality=100&vuk=-&ft=video" width="800"/>
</div>

## <img title="" src="https://user-images.githubusercontent.com/48054808/157800467-2a9946ad-30d1-49a9-b9db-ba33413d9c90.png" alt="" width="20"> Exchanges

- If you have any question or suggestion, please give us your valuable input via [GitHub Issues](https://github.com/PaddlePaddle/PaddleDetection/issues) 
  
  Welcome to join PaddleDetection user groups on QQ, WeChat (scan the QR code, add and reply "Detection" to the assistant)
  
  <div align="center">
  <img src="https://user-images.githubusercontent.com/48054808/157800129-2f9a0b72-6bb8-4b10-8310-93ab1639253f.jpg"  width = "200" />  
  <img src="https://user-images.githubusercontent.com/48054808/160531099-9811bbe6-cfbb-47d5-8bdb-c2b40684d7dd.png"  width = "200" />  
  </div>

## <img src="https://user-images.githubusercontent.com/48054808/157827140-03ffaff7-7d14-48b4-9440-c38986ea378c.png" width="20"/> Kit Structure

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
        <details><summary><b>Object Detection</b></summary>
          <ul>
            <li>Faster RCNN</li>
            <li>FPN</li>
            <li>Cascade-RCNN</li>
            <li>PSS-Det</li>
            <li>RetinaNet</li>
            <li>YOLOv3</li>  
            <li>PP-YOLOv1/v2</li>
            <li>PP-YOLO-Tiny</li>
            <li>PP-YOLOE</li>
            <li>YOLOX</li>
            <li>SSD</li>
            <li>CenterNet</li>
            <li>FCOS</li>  
            <li>TTFNet</li>
            <li>TOOD</li>
            <li>GFL</li>
            <li>PP-PicoDet</li>
            <li>DETR</li>
            <li>Deformable DETR</li>
            <li>Swin Transformer</li>
            <li>Sparse RCNN</li>
         </ul></details>
        <details><summary><b>Instance Segmentation</b></summary>
         <ul>
            <li>Mask RCNN</li>
            <li>Cascade Mask RCNN</li>
            <li>SOLOv2</li>
        </ul></details>
        <details><summary><b>Face Detection</b></summary>
        <ul>
            <li>BlazeFace</li>
        </ul></details>
        <details><summary><b>Multi-Object-Tracking</b></summary>
        <ul>
            <li>JDE</li>
            <li>FairMOT</li>
            <li>DeepSORT</li>
            <li>ByteTrack</li>
        </ul></details>
        <details><summary><b>KeyPoint-Detection</b></summary>
        <ul>
            <li>HRNet</li>
            <li>HigherHRNet</li>
            <li>Lite-HRNet</li>
            <li>PP-TinyPose</li>
        </ul></details>
      </ul>
      </td>
      <td>
        <details><summary><b>Details</b></summary>
        <ul>
          <li>ResNet(&vd)</li>
          <li>Res2Net(&vd)</li>
          <li>CSPResNet</li>
          <li>SENet</li>
          <li>Res2Net</li>
          <li>HRNet</li>
          <li>Lite-HRNet</li>
          <li>DarkNet</li>
          <li>CSPDarkNet</li>
          <li>MobileNetv1/v3</li>  
          <li>ShuffleNet</li>
          <li>GhostNet</li>
          <li>BlazeNet</li>
          <li>DLA</li>
          <li>HardNet</li>
          <li>LCNet</li>  
          <li>ESNet</li>  
          <li>Swin-Transformer</li>
        </ul></details>
      </td>
      <td>
        <details><summary><b>Common</b></summary>
          <ul>
            <li>Sync-BN</li>
            <li>Group Norm</li>
            <li>DCNv2</li>
            <li>EMA</li>
          </ul> </details>
        </ul>
        <details><summary><b>KeyPoint</b></summary>
          <ul>
            <li>DarkPose</li>
          </ul></details>
        </ul>
        <details><summary><b>FPN</b></summary>
          <ul>
            <li>BiFPN</li>
            <li>CSP-PAN</li>
            <li>Custom-PAN</li>
            <li>ES-PAN</li>
            <li>HRFPN</li>
          </ul> </details>
        </ul>  
        <details><summary><b>Loss</b></summary>
          <ul>
            <li>Smooth-L1</li>
            <li>GIoU/DIoU/CIoU</li>  
            <li>IoUAware</li>
            <li>Focal Loss</li>
            <li>CT Focal Loss</li>
            <li>VariFocal Loss</li>
          </ul> </details>
        </ul>  
        <details><summary><b>Post-processing</b></summary>
          <ul>
            <li>SoftNMS</li>
            <li>MatrixNMS</li>  
          </ul> </details>  
        </ul>
        <details><summary><b>Speed</b></summary>
          <ul>
            <li>FP16 training</li>
            <li>Multi-machine training </li>  
          </ul> </details>  
        </ul>  
      </td>
      <td>
        <details><summary><b>Details</b></summary>
        <ul>
          <li>Resize</li>  
          <li>Lighting</li>  
          <li>Flipping</li>  
          <li>Expand</li>
          <li>Crop</li>
          <li>Color Distort</li>  
          <li>Random Erasing</li>  
          <li>Mixup </li>
          <li>AugmentHSV</li>
          <li>Mosaic</li>
          <li>Cutmix </li>
          <li>Grid Mask</li>
          <li>Auto Augment</li>  
          <li>Random Perspective</li>  
        </ul> </details>  
      </td>  
    </tr>

</td>
    </tr>
  </tbody>
</table>

## <img src="https://user-images.githubusercontent.com/48054808/157801371-9a9a8c65-1690-4123-985a-e0559a7f9494.png" width="20"/> Model Performance

<details>
<summary><b> Performance comparison of Cloud models</b></summary>

The comparison between COCO mAP and FPS on Tesla V100 of representative models of each architectures and backbones.

<div align="center">
  <img src="docs/images/fps_map.png" />
</div>

**Clarification：**

- `CBResNet` stands for `Cascade-Faster-RCNN-CBResNet200vd-FPN`, which has highest mAP on COCO as 53.3%
- `Cascade-Faster-RCNN`stands for `Cascade-Faster-RCNN-ResNet50vd-DCN`, which has been optimized to 20 FPS inference speed when COCO mAP as 47.8% in PaddleDetection models
- `PP-YOLO` reached accuracy as 45.9% on COCO dataset, inference speed as 72.9 FPS on Tesla V100, higher than [YOLOv4](https://arxiv.org/abs/2004.10934) in terms of speed and accuracy
- `PP-YOLO v2`are optimized `PP-YOLO`. It reached accuracy as 49.5% on COCO dataset, inference speed as 68.9 FPS on Tesla V100.
- `PP-YOLOE`are optimized `PP-YOLO v2`. It reached accuracy as 51.4% on COCO dataset, inference speed as 78.1 FPS on Tesla V100
- The models in the figure are available in the[ model library](#模型库)

</details>

<details>
<summary><b> Performance omparison on mobiles</b></summary>

The comparison between COCO mAP and FPS on Qualcomm Snapdragon 865 processor of models on mobile devices.

<div align="center">
  <img src="docs/images/mobile_fps_map.png" width=600/>
</div>

**Clarification：**

- Tests were conducted on Qualcomm Snapdragon 865 (4 \*A77 + 4 \*A55) batch_size=1, 4 thread, and NCNN inference library, test script see [MobileDetBenchmark](https://github.com/JiweiMaster/MobileDetBenchmark)
- [PP-PicoDet](configs/picodet) and [PP-YOLO-Tiny](configs/ppyolo) are self-developed models of PaddleDetection, and other models are not tested yet.

</details>

## <img src="https://user-images.githubusercontent.com/48054808/157829890-a535b8a6-631c-4c87-b861-64d4b32b2d6a.png" width="20"/> Model libraries

<details>
<summary><b> 1. General detection</b></summary>

#### PP-YOLOE series Recommended scenarios: Cloud GPU such as Nvidia V100, T4 and edge devices such as Jetson series

| Model      | COCO Accuracy（mAP） | V100 TensorRT FP16 Speed(FPS) | Configuration                                           | Download                                                                                 |
|:---------- |:------------------:|:-----------------------------:|:-------------------------------------------------------:|:----------------------------------------------------------------------------------------:|
| PP-YOLOE-s | 42.7               | 333.3                         | [Link](configs/ppyoloe/ppyoloe_crn_l_300e_coco.yml)     | [Download](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams)      |
| PP-YOLOE-m | 48.6               | 208.3                         | [Link](configs/ppyolo/ppyolo_r50vd_dcn_2x_coco.yml)     | [Download](https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_2x_coco.pdparams)     |
| PP-YOLOE-l | 50.9               | 149.2                         | [Link](configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml) | [Download](https://paddledet.bj.bcebos.com/models/ppyolov2_r50vd_dcn_365e_coco.pdparams) |
| PP-YOLOE-x | 51.9               | 95.2                          | [Link](configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml) | [Download](https://paddledet.bj.bcebos.com/models/ppyolov2_r50vd_dcn_365e_coco.pdparams) |

#### PP-PicoDet series Recommended scenarios: Mobile chips and x86 CPU devices, such as ARM CPU(RK3399, Raspberry Pi) and NPU(BITMAIN)

| Model      | COCO Accuracy（mAP） | Snapdragon 865 four-thread speed (ms) | Configuration                                         | Download                                                                              |
|:---------- |:------------------:|:-------------------------------------:|:-----------------------------------------------------:|:-------------------------------------------------------------------------------------:|
| PicoDet-XS | 23.5               | 7.81                                  | [Link](configs/picodet/picodet_xs_320_coco_lcnet.yml) | [Download](https://paddledet.bj.bcebos.com/models/picodet_xs_320_coco_lcnet.pdparams) |
| PicoDet-S  | 29.1               | 9.56                                  | [Link](configs/picodet/picodet_s_320_coco_lcnet.yml)  | [Download](https://paddledet.bj.bcebos.com/models/picodet_s_320_coco_lcnet.pdparams)  |
| PicoDet-M  | 34.4               | 17.68                                 | [Link](configs/picodet/picodet_m_320_coco_lcnet.yml)  | [Download](https://paddledet.bj.bcebos.com/models/picodet_m_320_coco_lcnet.pdparams)  |
| PicoDet-L  | 36.1               | 25.21                                 | [Link](configs/picodet/picodet_l_320_coco_lcnet.yml)  | [Download](https://paddledet.bj.bcebos.com/models/picodet_l_320_coco_lcnet.pdparams)  |

#### Frontier detection algorithm

| Model    | COCO Accuracy（mAP） | V100 TensorRT FP16 speed(FPS) | Configuration                                                                                                  | Download                                                                       |
|:-------- |:------------------:|:-----------------------------:|:--------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------:|
| YOLOX-l  | 50.1               | 107.5                         | [Link](configs/yolox/yolox_l_300e_coco.yml)                                                                    | [Download](https://paddledet.bj.bcebos.com/models/yolox_l_300e_coco.pdparams)  |
| YOLOv5-l | 48.6               | 136.0                         | [Link](https://github.com/nemonameless/PaddleDetection_YOLOv5/blob/main/configs/yolov5/yolov5_l_300e_coco.yml) | [Download](https://paddledet.bj.bcebos.com/models/yolov5_l_300e_coco.pdparams) |

#### Other general purpose models [doc](docs/MODEL_ZOO_cn.md)

</details>

<details>
<summary><b> 2. Instance segmentation</b></summary>

| Model             | Introduction                                             | Recommended Scenarios                         | COCO Accuracy(mAP)               | Configuration                                                           | Download                                                                                              |
|:----------------- |:-------------------------------------------------------- |:--------------------------------------------- |:--------------------------------:|:-----------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------:|
| Mask RCNN         | Two-stage instance segmentation algorithm                | <div style="width: 50pt">Edge-Cloud end</div> | box AP: 41.4 <br/> mask AP: 37.5 | [Link](configs/mask_rcnn/mask_rcnn_r50_vd_fpn_2x_coco.yml)              | [Download](https://paddledet.bj.bcebos.com/models/mask_rcnn_r50_vd_fpn_2x_coco.pdparams)              |
| Cascade Mask RCNN | Two-stage instance segmentation algorithm                | <div style="width: 50pt">Edge-Cloud end</div> | box AP: 45.7 <br/> mask AP: 39.7 | [Link](configs/mask_rcnn/cascade_mask_rcnn_r50_vd_fpn_ssld_2x_coco.yml) | [Download](https://paddledet.bj.bcebos.com/models/cascade_mask_rcnn_r50_vd_fpn_ssld_2x_coco.pdparams) |
| SOLOv2            | Lightweight single-stage instance segmentation algorithm | <div style="width: 50pt">Edge-Cloud end</div> | mask AP: 38.0                    | [Link](configs/solov2/solov2_r50_fpn_3x_coco.yml)                       | [Download](https://paddledet.bj.bcebos.com/models/solov2_r50_fpn_3x_coco.pdparams)                    |

</details>

<details>
<summary><b> 3. Keypoint detection</b></summary>

| Model                | Introduction                                                                                  | Recommended scenarios                         | COCO Accuracy（AP） | Speed                             | Configuration                                             | Download                                                                                    |
|:-------------------- |:--------------------------------------------------------------------------------------------- |:--------------------------------------------- |:-----------------:|:---------------------------------:|:---------------------------------------------------------:|:-------------------------------------------------------------------------------------------:|
| HRNet-w32 + DarkPose | <div style="width: 130pt">Top-down Keypoint detection algorithm<br/>Input size: 384x288</div> | <div style="width: 50pt">Edge-Cloud end</div> | 78.3              | T4 TensorRT FP16 2.96ms           | [Link](configs/keypoint/hrnet/dark_hrnet_w32_384x288.yml) | [Download](https://paddledet.bj.bcebos.com/models/keypoint/dark_hrnet_w32_384x288.pdparams) |
| HRNet-w32 + DarkPose | Top-down Keypoint detection algorithm<br/>Input size: 256x192                                 | Edge-Cloud end                                | 78.0              | T4 TensorRT FP16 1.75ms           | [Link](configs/keypoint/hrnet/dark_hrnet_w32_256x192.yml) | [Download](https://paddledet.bj.bcebos.com/models/keypoint/dark_hrnet_w32_256x192.pdparams) |
| PP-TinyPose          | Light-weight keypoint algorithm<br/>Input size: 256x192                                       | Mobile                                        | 68.8              | Snapdragon 865 four-thread 6.30ms | [Link](configs/keypoint/tiny_pose/tinypose_256x192.yml)   | [Download](https://bj.bcebos.com/v1/paddledet/models/keypoint/tinypose_256x192.pdparams)    |
| PP-TinyPose          | Light-weight keypoint algorithm<br/>Input size: 128x96                                        | Mobile                                        | 58.1              | Snapdragon 865 four-thread 2.37ms | [Link](configs/keypoint/tiny_pose/tinypose_128x96.yml)    | [Download](https://bj.bcebos.com/v1/paddledet/models/keypoint/tinypose_128x96.pdparams)     |

#### Other keypoint detection models [doc](configs/keypoint)

</details>

<details>
<summary><b> 4. Multi-object tracking PP-Tracking</b></summary>

| Model     | Introduction                                                  | Recommended scenarios | Accuracy               | Configuration                                                           | Download                                                                                              |
|:--------- |:------------------------------------------------------------- |:--------------------- |:----------------------:|:-----------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------:|
| DeepSORT  | SDE Multi-object tracking algorithm, independent ReID models  | Edge-Cloud end        | MOT-17 half val:  66.9 | [Link](configs/mot/deepsort/deepsort_jde_yolov3_pcb_pyramid.yml)        | [Download](https://paddledet.bj.bcebos.com/models/mot/deepsort/deepsort_pcb_pyramid_r101.pdparams)    |
| ByteTrack | SDE Multi-object tracking algorithm with detection model only | Edge-Cloud end        | MOT-17 half val:  77.3 | [Link](configs/mot/bytetrack/detector/yolox_x_24e_800x1440_mix_det.yml) | [Download](https://paddledet.bj.bcebos.com/models/mot/deepsort/yolox_x_24e_800x1440_mix_det.pdparams) |
| JDE       | JDE multi-object tracking algorithm multi-task learning       | Edge-Cloud end        | MOT-16 test: 64.6      | [Link](configs/mot/jde/jde_darknet53_30e_1088x608.yml)                  | [Download](https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_1088x608.pdparams)            |
| FairMOT   | JDE multi-object tracking algorithm multi-task learning       | Edge-Cloud end        | MOT-16 test: 75.0      | [Link](configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml)              | [Download](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608.pdparams)            |

#### Other multi-object tracking models [docs](configs/mot)

</details>

<details>
<summary><b> 5. Industrial real-time pedestrain analysis tool-PP Human</b></summary>

| Function \ Model                     | Obejct detection                                                                       | Multi- object tracking                                                                 | Attribute recognition                                                                     | Keypoint detection                                                                        | Action recognition                                                | ReID                                                                   |
|:------------------------------------ |:-------------------------------------------------------------------------------------- |:-------------------------------------------------------------------------------------- |:-----------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------:|:-----------------------------------------------------------------:|:----------------------------------------------------------------------:|
| Pedestrian Detection                 | [✅](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip) |                                                                                        |                                                                                           |                                                                                           |                                                                   |                                                                        |
| Pedestrian Tracking                  |                                                                                        | [✅](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip) |                                                                                           |                                                                                           |                                                                   |                                                                        |
| Attribute Recognition (Image)        | [✅](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip) |                                                                                        | [✅](https://bj.bcebos.com/v1/paddledet/models/pipeline/strongbaseline_r50_30e_pa100k.zip) |                                                                                           |                                                                   |                                                                        |
| Attribute Recognition (Video)        |                                                                                        | [✅](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip) |                                                                                           |                                                                                           |                                                                   |                                                                        |
| Falling Detection                    |                                                                                        | [✅](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip) |                                                                                           | [✅](https://bj.bcebos.com/v1/paddledet/models/pipeline/strongbaseline_r50_30e_pa100k.zip) | [✅](https://bj.bcebos.com/v1/paddledet/models/pipeline/STGCN.zip) |                                                                        |
| ReID                                 |                                                                                        | [✅](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip) |                                                                                           |                                                                                           |                                                                   | [✅](https://bj.bcebos.com/v1/paddledet/models/pipeline/reid_model.zip) |
| **Accuracy**                         | mAP 56.3                                                                               | MOTA 72.0                                                                              | mA 94.86                                                                                  | AP 87.1                                                                                   | AP 96.43                                                          | mAP 98.8                                                               |
| **T4 TensorRT FP16 Inference speed** | 28.0ms                                                                                 | 33.1ms                                                                                 | Single person 2ms                                                                         | Single person 2.9ms                                                                       | Single person 2.7ms                                               | Single person 1.5ms                                                    |

</details>

**Click “ ✅ ” to download**

## <img src="https://user-images.githubusercontent.com/48054808/157828296-d5eb0ccb-23ea-40f5-9957-29853d7d13a9.png" width="20"/>Document tutorials

### Introductory tutorials

- [Installation](docs/tutorials/INSTALL_cn.md)
- [Quick start](docs/tutorials/QUICK_STARTED_cn.md)
- [Data preparation](docs/tutorials/data/README.md)
- [Geting Started on PaddleDetection](docs/tutorials/GETTING_STARTED_cn.md)
- [Customize data training](docs/tutorials/CustomizeDataTraining.md)
- [FAQ](docs/tutorials/FAQ)

### Advanced tutorials

- Configuration
  
  - [RCNN Configuration](docs/tutorials/config_annotation/faster_rcnn_r50_fpn_1x_coco_annotation.md)
  - [PP-YOLO Configuration](docs/tutorials/config_annotation/ppyolo_r50vd_dcn_1x_coco_annotation.md)

- Compression based on [PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim)
  
  - [Pruning/Quantization/Distillation Tutorial](configs/slim)

- [Inference deployment](deploy/README.md)
  
  - [Export model for inference](deploy/EXPORT_MODEL.md)
  
  - [Paddle Inference deployment](deploy/README.md)
    
    - [Inference deployment with Python](deploy/python)
    - [Inference deployment with C++](deploy/cpp)
  
  - [Paddle-Lite deployment](deploy/lite)
  
  - [Paddle Serving deployment](deploy/serving)
  
  - [ONNX model export](deploy/EXPORT_ONNX_MODEL.md)
  
  - [Inference benchmark](deploy/BENCHMARK_INFER.md)

- Advanced development
  
  - [Data processing module](docs/advanced_tutorials/READER.md)
  - [New object detection models](docs/advanced_tutorials/MODEL_TECHNICAL.md)
  - Custumization
    - [Object detection](docs/advanced_tutorials/customization/detection.md)
    - [Keypoint detection](docs/advanced_tutorials/customization/keypoint_detection.md)
    - [Multiple object tracking](docs/advanced_tutorials/customization/mot.md)
    - [Action recognition](docs/advanced_tutorials/customization/action.md)
    - [Attribute recognition](docs/advanced_tutorials/customization/attribute.md)

### Courses

- **[Theoretical foundation] [Object detection 7-day camp](https://aistudio.baidu.com/aistudio/education/group/info/1617):** Overview of object detection tasks, details of RCNN series object detection algorithm and YOLO series object detection algorithm, PP-YOLO optimization strategy and case sharing, introduction and practice of AnchorFree series algorithm

- **[Industrial application] [AI Fast Track industrial object detection technology and application](https://aistudio.baidu.com/aistudio/education/group/info/23670):** Super object detection algorithms, real-time pedestrian analysis system PP-Human, breakdown and practice of object detection industrial application

- **[Industrial features] 2022.3.26** **[Smart City Industry Seven-Day Class](https://aistudio.baidu.com/aistudio/education/group/info/25620)** : Urban planning, Urban governance, Smart governance service, Traffic management, community governance. 

### [Industrial tutorial examples](./industrial_tutorial/README_cn.md)

- [Road litter detection based on PP-PicoDet Plus](https://aistudio.baidu.com/aistudio/projectdetail/3561097)

- [Communication tower detection based on PP-PicoDet and deployment on Android](https://aistudio.baidu.com/aistudio/projectdetail/3561097)

- [Tile surface defect detection based on Faster-RCNN](https://aistudio.baidu.com/aistudio/projectdetail/2571419)

- [PCB defect detection based on PaddleDetection](https://aistudio.baidu.com/aistudio/projectdetail/2367089)

- [Visitor flow statistics based on FairMOT](https://aistudio.baidu.com/aistudio/projectdetail/2421822)

- [Falling detection based on YOLOv3](https://aistudio.baidu.com/aistudio/projectdetail/2500639)

- [Compliance detection based on human key  point detection](https://aistudio.baidu.com/aistudio/projectdetail/4061642?contributionType=1)

## <img title="" src="https://user-images.githubusercontent.com/48054808/157836473-1cf451fa-f01f-4148-ba68-b6d06d5da2f9.png" alt="" width="20"> Applications

- [Fitness app on android mobile](https://github.com/zhiboniu/pose_demo_android)
- [PP-Tracking GUI Visualization Interface](https://github.com/yangyudong2020/PP-Tracking_GUi)

## Recommended third-party tutorials

- [Deployment of PaddleDetection for Windows I ](https://zhuanlan.zhihu.com/p/268657833)
- [Deployment of PaddleDetection for Windows II](https://zhuanlan.zhihu.com/p/280206376)
- [Deployment of PaddleDetection on Jestson Nano](https://zhuanlan.zhihu.com/p/319371293) 
- [How to deploy YOLOv3 model on Raspberry Pi for Helmet detection](https://github.com/PaddleCV-FAQ/PaddleDetection-FAQ/blob/main/Lite%E9%83%A8%E7%BD%B2/yolov3_for_raspi.md)
- [Use SSD-MobileNetv1 for a project -- From dataset to deployment on Raspberry Pi](https://github.com/PaddleCV-FAQ/PaddleDetection-FAQ/blob/main/Lite%E9%83%A8%E7%BD%B2/ssd_mobilenet_v1_for_raspi.md)

## <img src="https://user-images.githubusercontent.com/48054808/157835981-ef6057b4-6347-4768-8fcc-cd07fcc3d8b0.png" width="20"/> Version updates

Please refer to the[ Release note ](https://github.com/PaddlePaddle/Paddle/wiki/PaddlePaddle-2.3.0-Release-Note-EN)for more details about the updates

## <img title="" src="https://user-images.githubusercontent.com/48054808/157835345-f5d24128-abaf-4813-b793-d2e5bdc70e5a.png" alt="" width="20">  License

PaddlePaddle is provided under the [Apache 2.0 license](LICENSE)

## <img src="https://user-images.githubusercontent.com/48054808/157835796-08d4ffbc-87d9-4622-89d8-cf11a44260fc.png" width="20"/> Contribute your code

We appreciate your contributions and your feedback！

- Thank [Mandroide](https://github.com/Mandroide) for code cleanup and 
- Thank [FL77N](https://github.com/FL77N/) for `Sparse-RCNN`model
- Thank [Chen-Song](https://github.com/Chen-Song) for `Swin Faster-RCNN`model
- Thank [yangyudong](https://github.com/yangyudong2020), [hchhtc123](https://github.com/hchhtc123) for developing PP-Tracking GUI interface
- Thank [Shigure19](https://github.com/Shigure19) for developing PP-TinyPose fitness APP
- Thank [manangoel99](https://github.com/manangoel99) for Wandb visualization methods

## <img src="https://user-images.githubusercontent.com/48054808/157835276-9aab9d1c-1c46-446b-bdd4-5ab75c5cfa48.png" width="20"/> Quote

```
@misc{ppdet2019,
title={PaddleDetection, Object detection and instance segmentation toolkit based on PaddlePaddle.},
author={PaddlePaddle Authors},
howpublished = {\url{https://github.com/PaddlePaddle/PaddleDetection}},
year={2019}
}
```
