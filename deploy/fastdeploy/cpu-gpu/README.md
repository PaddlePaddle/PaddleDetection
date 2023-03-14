[English](README.md) | 简体中文

# PaddleDetection高性能全场景模型部署方案—FastDeploy

## 1. 说明
PaddleDetection支持利用FastDeploy在NVIDIA GPU、X86 CPU、飞腾CPU、ARM CPU、Intel GPU(独立显卡/集成显卡)硬件上快速部署检测模型

## 2. 使用预导出的模型列表  
为了方便开发者的测试，下面提供了PaddleDetection导出的各系列模型，开发者可直接下载使用。其中精度指标来源于PaddleDetection中对各模型的介绍，详情各参考PaddleDetection中的说明。

| 模型                                                               | 参数大小    | 精度    | 备注 |
|:---------------------------------------------------------------- |:----- |:----- | :------ |
| [picodet_l_320_coco_lcnet](https://bj.bcebos.com/paddlehub/fastdeploy/picodet_l_320_coco_lcnet.tgz) |23MB | Box AP 42.6% |
| [ppyoloe_crn_l_300e_coco](https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco.tgz) |200MB | Box AP 51.4% |
| [ppyoloe_plus_crn_m_80e_coco](https://bj.bcebos.com/fastdeploy/models/ppyoloe_plus_crn_m_80e_coco.tgz) |83.3MB | Box AP 49.8% |
| [ppyolo_r50vd_dcn_1x_coco](https://bj.bcebos.com/paddlehub/fastdeploy/ppyolo_r50vd_dcn_1x_coco.tgz) | 180MB | Box AP 44.8% | 暂不支持TensorRT |
| [ppyolov2_r101vd_dcn_365e_coco](https://bj.bcebos.com/paddlehub/fastdeploy/ppyolov2_r101vd_dcn_365e_coco.tgz) | 282MB | Box AP 49.7% | 暂不支持TensorRT |
| [yolov3_darknet53_270e_coco](https://bj.bcebos.com/paddlehub/fastdeploy/yolov3_darknet53_270e_coco.tgz) |237MB | Box AP 39.1% | |
| [yolox_s_300e_coco](https://bj.bcebos.com/paddlehub/fastdeploy/yolox_s_300e_coco.tgz) | 35MB | Box AP 40.4% | |
| [faster_rcnn_r50_vd_fpn_2x_coco](https://bj.bcebos.com/paddlehub/fastdeploy/faster_rcnn_r50_vd_fpn_2x_coco.tgz) | 160MB | Box AP 40.8%| 暂不支持TensorRT |
| [mask_rcnn_r50_1x_coco](https://bj.bcebos.com/paddlehub/fastdeploy/mask_rcnn_r50_1x_coco.tgz) | 128M | Box AP 37.4%, Mask AP 32.8%| 暂不支持TensorRT、ORT |
| [ssd_mobilenet_v1_300_120e_voc](https://bj.bcebos.com/paddlehub/fastdeploy/ssd_mobilenet_v1_300_120e_voc.tgz) | 24.9M | Box AP 73.8%| 暂不支持TensorRT、ORT |
| [ssd_vgg16_300_240e_voc](https://bj.bcebos.com/paddlehub/fastdeploy/ssd_vgg16_300_240e_voc.tgz) | 106.5M | Box AP 77.8%| 暂不支持TensorRT、ORT |
| [ssdlite_mobilenet_v1_300_coco](https://bj.bcebos.com/paddlehub/fastdeploy/ssdlite_mobilenet_v1_300_coco.tgz) | 29.1M | | 暂不支持TensorRT、ORT |
| [rtmdet_l_300e_coco](https://bj.bcebos.com/paddlehub/fastdeploy/rtmdet_l_300e_coco.tgz) | 224M | Box AP 51.2%|  |
| [rtmdet_s_300e_coco](https://bj.bcebos.com/paddlehub/fastdeploy/rtmdet_s_300e_coco.tgz) | 42M | Box AP 44.5%|  |
| [yolov5_l_300e_coco](https://bj.bcebos.com/paddlehub/fastdeploy/yolov5_l_300e_coco.tgz) | 183M | Box AP 48.9%|  |
| [yolov5_s_300e_coco](https://bj.bcebos.com/paddlehub/fastdeploy/yolov5_s_300e_coco.tgz) | 31M | Box AP 37.6%|  |
| [yolov6_l_300e_coco](https://bj.bcebos.com/paddlehub/fastdeploy/yolov6_l_300e_coco.tgz) | 229M | Box AP 51.0%|  |
| [yolov6_s_400e_coco](https://bj.bcebos.com/paddlehub/fastdeploy/yolov6_s_400e_coco.tgz) | 68M | Box AP 43.4%|  |
| [yolov7_l_300e_coco](https://bj.bcebos.com/paddlehub/fastdeploy/yolov7_l_300e_coco.tgz) | 145M | Box AP 51.0%|  |
| [yolov7_x_300e_coco](https://bj.bcebos.com/paddlehub/fastdeploy/yolov7_x_300e_coco.tgz) | 277M | Box AP 53.0%|  |
| [cascade_rcnn_r50_fpn_1x_coco](https://bj.bcebos.com/paddlehub/fastdeploy/cascade_rcnn_r50_fpn_1x_coco.tgz) | 271M | Box AP 41.1%|  暂不支持TensorRT、ORT |
| [cascade_rcnn_r50_vd_fpn_ssld_2x_coco](https://bj.bcebos.com/paddlehub/fastdeploy/cascade_rcnn_r50_vd_fpn_ssld_2x_coco.tgz) | 271M | Box AP 45.0%|  暂不支持TensorRT、ORT |
| [faster_rcnn_enhance_3x_coco](https://bj.bcebos.com/paddlehub/fastdeploy/faster_rcnn_enhance_3x_coco.tgz) | 119M | Box AP 41.5%|  暂不支持TensorRT、ORT |
| [fcos_r50_fpn_1x_coco](https://bj.bcebos.com/paddlehub/fastdeploy/fcos_r50_fpn_1x_coco.tgz) | 129M | Box AP 39.6%|  暂不支持TensorRT |
| [gfl_r50_fpn_1x_coco](https://bj.bcebos.com/paddlehub/fastdeploy/gfl_r50_fpn_1x_coco.tgz) | 128M | Box AP 41.0%|  暂不支持TensorRT |
| [ppyoloe_crn_l_80e_sliced_visdrone_640_025](https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_80e_sliced_visdrone_640_025.tgz) | 200M | Box AP 31.9%|  |
| [retinanet_r101_fpn_2x_coco](https://bj.bcebos.com/paddlehub/fastdeploy/retinanet_r101_fpn_2x_coco.tgz) | 210M | Box AP 40.6%|  暂不支持TensorRT、ORT |
| [retinanet_r50_fpn_1x_coco](https://bj.bcebos.com/paddlehub/fastdeploy/retinanet_r50_fpn_1x_coco.tgz) | 136M | Box AP 37.5%|  暂不支持TensorRT、ORT |
| [tood_r50_fpn_1x_coco](https://bj.bcebos.com/paddlehub/fastdeploy/tood_r50_fpn_1x_coco.tgz) | 130M | Box AP 42.5%|  暂不支持TensorRT、ORT |
| [ttfnet_darknet53_1x_coco](https://bj.bcebos.com/paddlehub/fastdeploy/ttfnet_darknet53_1x_coco.tgz) | 178M | Box AP 33.5%|  暂不支持TensorRT、ORT |  
| [yolov8_x_500e_coco](https://bj.bcebos.com/paddlehub/fastdeploy/yolov8_x_500e_coco.tgz) | 265M | Box AP 53.8%
| [yolov8_l_500e_coco](https://bj.bcebos.com/paddlehub/fastdeploy/yolov8_l_500e_coco.tgz) | 173M | Box AP 52.8%
| [yolov8_m_500e_coco](https://bj.bcebos.com/paddlehub/fastdeploy/yolov8_m_500e_coco.tgz) | 99M | Box AP 50.2%
| [yolov8_s_500e_coco](https://bj.bcebos.com/paddlehub/fastdeploy/yolov8_s_500e_coco.tgz) | 43M | Box AP 44.9%
| [yolov8_n_500e_coco](https://bj.bcebos.com/paddlehub/fastdeploy/yolov8_n_500e_coco.tgz) | 13M | Box AP 37.3%


## 3. 自行导出PaddleDetection部署模型  
### 3.1 模型版本
支持[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)大于等于2.4版本的PaddleDetection模型部署。目前FastDeploy测试过成功部署的模型:   

- [PP-YOLOE(含PP-YOLOE+)系列模型](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/ppyoloe)
- [PicoDet系列模型](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/picodet)
- [PP-YOLO系列模型(含v2)](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/ppyolo)
- [YOLOv3系列模型](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/yolov3)
- [YOLOX系列模型](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/yolox)
- [FasterRCNN系列模型](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/faster_rcnn)
- [MaskRCNN系列模型](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/mask_rcnn)
- [SSD系列模型](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/ssd)
- [YOLOv5系列模型](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.6/configs/yolov5)
- [YOLOv6系列模型](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.6/configs/yolov6)
- [YOLOv7系列模型](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.6/configs/yolov7)  
- [YOLOv8系列模型](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.6/configs/yolov8)
- [RTMDet系列模型](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.6/configs/rtmdet)
- [CascadeRCNN系列模型](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/cascade_rcnn)
- [PSSDet系列模型](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/rcnn_enhance)
- [RetinaNet系列模型](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/retinanet)
- [PPYOLOESOD系列模型](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/smalldet)
- [FCOS系列模型](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/fcos)
- [TTFNet系列模型](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/ttfnet)
- [TOOD系列模型](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/tood)
- [GFL系列模型](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/gfl)

### 3.2 模型导出   
PaddleDetection模型导出，请参考其文档说明[模型导出](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.6/deploy/EXPORT_MODEL.md)，**注意**：PaddleDetection导出的模型包含`model.pdmodel`、`model.pdiparams`和`infer_cfg.yml`三个文件，FastDeploy会从yaml文件中获取模型在推理时需要的预处理信息

### 3.3 导出须知  
如果您是自行导出PaddleDetection推理模型，请注意以下问题：
- 在导出模型时不要进行NMS的去除操作，正常导出即可  
- 如果用于跑原生TensorRT后端（非Paddle Inference后端），不要添加--trt参数
- 导出模型时，不要添加`fuse_normalize=True`参数

## 4. 详细的部署示例  
- [Python部署](python)
- [C++部署](cpp)