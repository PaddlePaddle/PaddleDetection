## 汇总信息

已填写的部分表示可以使用本工具进行一键测试，未填写的表示正在支持中。

**字段说明：**
- 基础训练预测：包括模型训练、Paddle Inference Python预测。
- 更多训练方式：包括多机多卡、混合精度。
- 模型压缩：包括裁剪、离线/在线量化、蒸馏。
- 其他预测部署：包括Paddle Inference C++预测、Paddle Serving部署、Paddle-Lite部署等。

| 算法论文 | 模型名称 | 模型类型 | 基础<br>训练预测 | 更多<br>训练方式 | 模型压缩 |  其他预测部署  |
| :--- | :--- | :----: | :--------: | :---- | :---- | :---- |
| [YOLOv3](https://arxiv.org/abs/1804.02767) | [yolov3_darknet53_270e_coco](../../configs/yolov3/yolov3_darknet53_270e_coco.yml) | 目标检测 | 支持 | 混合精度 | FPGM裁剪 <br> PACT量化 <br> 离线量化 | Paddle Inference: C++  |
| YOLOv3 | [yolov3_mobilenet_v1_270e_coco](../../configs/yolov3/yolov3_mobilenet_v1_270e_coco.yml) | 目标检测 | 支持 | 混合精度 |  | Paddle Inference: C++  |
| YOLOv3 | [yolov3_mobilenet_v3_large_270e_coco](../../configs/yolov3/yolov3_mobilenet_v3_large_270e_coco.yml) | 目标检测 | 支持 | 混合精度 |  | Paddle Inference: C++  |
| YOLOv3 | [yolov3_r34_270e_coco](../../configs/yolov3/yolov3_r34_270e_coco.yml) | 目标检测 | 支持 | 混合精度 |  | Paddle Inference: C++  |
| YOLOv3 | [yolov3_r50vd_dcn_270e_coco](../../configs/yolov3/yolov3_r50vd_dcn_270e_coco.yml) | 目标检测 | 支持 | 混合精度 |  | Paddle Inference: C++  |
| [PPYOLO](https://arxiv.org/abs/2007.12099) | [ppyolo_mbv3_large_coco](../../configs/ppyolo/ppyolo_mbv3_large_coco.yml) | 目标检测  | 支持 | 混合精度 | FPGM裁剪 <br> PACT量化 <br> 离线量化 | Paddle Inference: C++  |
| PPYOLO | [ppyolo_r50vd_dcn_1x_coco](../../configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml) | 目标检测  | 支持 | 混合精度 | FPGM裁剪 <br> PACT量化 <br> 离线量化 | Paddle Inference: C++  |
| PPYOLO | [ppyolo_mbv3_small_coco](../../configs/ppyolo/ppyolo_mbv3_small_coco.yml) | 目标检测  | 支持 | 混合精度 |  | Paddle Inference: C++  |
| PPYOLO | [ppyolo_r18vd_coco](../../configs/ppyolo/ppyolo_r18vd_coco.yml) | 目标检测  | 支持 | 混合精度 |  | Paddle Inference: C++  |
| PPYOLO-tiny | [ppyolo_tiny_650e_coco](../../configs/ppyolo/ppyolo_tiny_650e_coco.yml) | 目标检测 | 支持 | 混合精度 |  | Paddle Inference: C++  |
| [PPYOLOv2](https://arxiv.org/abs/2104.10419) | [ppyolov2_r50vd_dcn_365e_coco](../../configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml) | 目标检测  | 支持 | 多机多卡 <br> 混合精度 |  | Paddle Inference: C++  |
| PPYOLOv2 | [ppyolov2_r50vd_dcn_365e_coco](../../configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml) | 目标检测  | 支持 | 混合精度 |  | Paddle Inference: C++  |
| PPYOLOv2 | [ppyolov2_r101vd_dcn_365e_coco](../../configs/ppyolo/ppyolov2_r101vd_dcn_365e_coco.yml) | 目标检测  | 支持 | 混合精度 |  | Paddle Inference: C++  |
| [PP-PicoDet](https://arxiv.org/abs/2111.00902) | [picodet_s_320_coco](../../configs/picodet/picodet_s_320_coco.yml) | 目标检测 | 支持 | 混合精度 |  | Paddle Inference: C++  |
| PP-PicoDet | [picodet_m_416_coco](../../configs/picodet/picodet_m_416_coco.yml) | 目标检测 | 支持 | 混合精度 |  | Paddle Inference: C++  |
| PP-PicoDet | [picodet_l_640_coco](../../configs/picodet/picodet_l_640_coco.yml) | 目标检测 | 支持 | 混合精度 |  | Paddle Inference: C++  |
| PP-PicoDet | [picodet_lcnet_1_5x_416_coco](../../configs/picodet/more_config/picodet_lcnet_1_5x_416_coco.yml) | 目标检测 | 支持 | 混合精度 |  | Paddle Inference: C++  |
| PP-PicoDet | [picodet_mobilenetv3_large_1x_416_coco](../../configs/picodet/more_config/picodet_mobilenetv3_large_1x_416_coco.yml) | 目标检测 | 支持 | 混合精度 |  | Paddle Inference: C++  |
| PP-PicoDet | [picodet_r18_640_coco](../../configs/picodet/more_config/picodet_r18_640_coco.yml) | 目标检测 | 支持 | 混合精度 |  | Paddle Inference: C++  |
| PP-PicoDet | [picodet_shufflenetv2_1x_416_coco](../../configs/picodet/more_config/picodet_shufflenetv2_1x_416_coco.yml) | 目标检测 | 支持 | 混合精度 |  | Paddle Inference: C++  |
| [SSD](https://arxiv.org/abs/1512.02325) | [ssdlite_mobilenet_v1_300_coco](../../configs/ssd/ssdlite_mobilenet_v1_300_coco.yml) | 目标检测 | 支持 | 混合精度 |  | Paddle Inference: C++  |
| [Faster R-CNN](https://arxiv.org/abs/1506.01497) | [faster_rcnn_r50_fpn_1x_coco](../../configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml) | 目标检测 | 支持 | 混合精度 |  | Paddle Inference: C++  |
| Faster R-CNN | [faster_rcnn_r34_fpn_1x_coco](../../configs/faster_rcnn/faster_rcnn_r34_fpn_1x_coco.yml) | 目标检测 | 支持 | 混合精度 |  | Paddle Inference: C++  |
| Faster R-CNN | [faster_rcnn_r34_vd_fpn_1x_coco](../../configs/faster_rcnn/faster_rcnn_r34_vd_fpn_1x_coco.yml) | 目标检测 | 支持 | 混合精度 |  | Paddle Inference: C++  |
| Faster R-CNN | [faster_rcnn_r50_1x_coco](../../configs/faster_rcnn/faster_rcnn_r50_1x_coco.yml) | 目标检测 | 支持 | 混合精度 |  | Paddle Inference: C++  |
| Faster R-CNN | [faster_rcnn_r50_vd_1x_coco](../../configs/faster_rcnn/faster_rcnn_r50_vd_1x_coco.yml) | 目标检测 | 支持 | 混合精度 |  | Paddle Inference: C++  |
| Faster R-CNN | [faster_rcnn_r50_vd_fpn_1x_coco](../../configs/faster_rcnn/faster_rcnn_r50_vd_fpn_1x_coco.yml) | 目标检测 | 支持 | 混合精度 |  | Paddle Inference: C++  |
| Faster R-CNN | [faster_rcnn_r101_1x_coco](../../configs/faster_rcnn/faster_rcnn_r101_1x_coco.yml) | 目标检测 | 支持 | 混合精度 |  | Paddle Inference: C++  |
| Faster R-CNN | [faster_rcnn_r101_fpn_1x_coco](../../configs/faster_rcnn/faster_rcnn_r101_fpn_1x_coco.yml) | 目标检测 | 支持 | 混合精度 |  | Paddle Inference: C++  |
| Faster R-CNN | [faster_rcnn_r101_vd_fpn_1x_coco](../../configs/faster_rcnn/faster_rcnn_r101_vd_fpn_1x_coco.yml) | 目标检测 | 支持 | 混合精度 |  | Paddle Inference: C++  |
| Faster R-CNN | [faster_rcnn_x101_vd_64x4d_fpn_1x_coco](../../configs/faster_rcnn/faster_rcnn_x101_vd_64x4d_fpn_1x_coco.yml) | 目标检测 | 支持 | 混合精度 |  | Paddle Inference: C++  |
| Faster R-CNN | [faster_rcnn_swin_tiny_fpn_1x_coco](../../configs/faster_rcnn/faster_rcnn_swin_tiny_fpn_1x_coco.yml) | 目标检测 | 支持 | 混合精度 |  | Paddle Inference: C++  |
| [Cascade Faster R-CNN](https://arxiv.org/abs/1712.00726) | [cascade_rcnn_r50_fpn_1x_coco](../../configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.yml) | 目标检测 | 支持 | 混合精度 |  | Paddle Inference: C++  |
| Cascade Faster R-CNN | [cascade_rcnn_r50_vd_fpn_ssld_1x_coco](../../configs/cascade_rcnn/cascade_rcnn_r50_vd_fpn_ssld_1x_coco.yml) | 目标检测 | 支持 | 混合精度 |  | Paddle Inference: C++  |
| [FCOS](https://arxiv.org/abs/1904.01355) | [fcos_r50_fpn_1x_coco](../../configs/fcos/fcos_r50_fpn_1x_coco.yml) | 目标检测 | 支持 | 混合精度 |  | Paddle Inference: C++  |
| FCOS | [fcos_dcn_r50_fpn_1x_coco](../../configs/fcos/fcos_dcn_r50_fpn_1x_coco.yml) | 目标检测 | 支持 | 混合精度 |  | Paddle Inference: C++  |
| [TTFNet](https://arxiv.org/abs/1909.00700) | [ttfnet_darknet53_1x_coco](../../configs/ttfnet/ttfnet_darknet53_1x_coco.yml) | 目标检测 | 支持 | 混合精度 |  | Paddle Inference: C++  |
| [S2ANet](https://arxiv.org/abs/2008.09397) | [s2anet_conv_2x_dota](../../configs/dota/s2anet_conv_2x_dota.yml) | 目标检测 | 支持 | 混合精度 |  | Paddle Inference: C++  |
| S2ANet | [s2anet_1x_spine](../../configs/dota/s2anet_1x_spine.yml) | 目标检测 | 支持 | 混合精度 |  | Paddle Inference: C++  |
| S2ANet | [s2anet_alignconv_2x_dota](../../configs/dota/s2anet_alignconv_2x_dota.yml) | 目标检测 | 支持 | 混合精度 |  | Paddle Inference: C++  |
| [BlazeFace](https://arxiv.org/abs/1907.05047) | [blazeface_1000e](../../configs/face_detection/blazeface_1000e.yml) | 目标检测 | 支持 | 混合精度 |  | Paddle Inference: C++  |
| BlazeFace | [blazeface_fpn_ssh_1000e](../../configs/face_detection/blazeface_fpn_ssh_1000e.yml) | 目标检测 | 支持 | 混合精度 |  | Paddle Inference: C++  |
| [Mask R-CNN](https://arxiv.org/abs/1703.06870) | [mask_rcnn_r50_fpn_1x_coco](../../configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.yml) | 实例分割 | 支持 | 混合精度 |  | Paddle Inference: C++  |
| Mask R-CNN | [mask_rcnn_r50_1x_coco](../../configs/mask_rcnn/mask_rcnn_r50_1x_coco.yml) | 实例分割 | 支持 | 混合精度 |  | Paddle Inference: C++  |
| Mask R-CNN | [mask_rcnn_r50_vd_fpn_1x_coco](../../configs/mask_rcnn/mask_rcnn_r50_vd_fpn_1x_coco.yml) | 实例分割 | 支持 | 混合精度 |  | Paddle Inference: C++  |
| Mask R-CNN | [mask_rcnn_r101_fpn_1x_coco](../../configs/mask_rcnn/mask_rcnn_r101_fpn_1x_coco.yml) | 实例分割 | 支持 | 混合精度 |  | Paddle Inference: C++  |
| Mask R-CNN | [mask_rcnn_r101_vd_fpn_1x_coco](../../configs/mask_rcnn/mask_rcnn_r101_vd_fpn_1x_coco.yml) | 实例分割 | 支持 | 混合精度 |  | Paddle Inference: C++  |
| Mask R-CNN | [mask_rcnn_x101_vd_64x4d_fpn_1x_coco](../../configs/mask_rcnn/mask_rcnn_x101_vd_64x4d_fpn_1x_coco.yml) | 实例分割 | 支持 | 混合精度 |  | Paddle Inference: C++  |
| [Cascade Mask R-CNN](https://arxiv.org/abs/1906.09756) | [cascade_mask_rcnn_r50_fpn_1x_coco](../../configs/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.yml) | 实例分割 | 支持 | 混合精度 |  | Paddle Inference: C++  |
| Cascade Mask R-CNN | [cascade_mask_rcnn_r50_vd_fpn_ssld_1x_coco](../../configs/cascade_rcnn/cascade_mask_rcnn_r50_vd_fpn_ssld_1x_coco.yml) | 实例分割 | 支持 | 混合精度 |  | Paddle Inference: C++  |
| [SOLOv2](https://arxiv.org/abs/2003.10152) | [solov2_r50_fpn_1x_coco](../../configs/solov2/solov2_r50_fpn_1x_coco.yml) | 实例分割 | 支持 | 混合精度 |  | Paddle Inference: C++  |
| SOLOv2 | [solov2_r50_enhance_coco](../../configs/solov2/solov2_r50_enhance_coco.yml) | 实例分割 | 支持 | 混合精度 |  | Paddle Inference: C++  |
| SOLOv2 | [solov2_r101_vd_fpn_3x_coco](../../configs/solov2/solov2_r101_vd_fpn_3x_coco.yml) | 实例分割 | 支持 | 混合精度 |  | Paddle Inference: C++  |
| [PP-Tinypose] | [tinypose_128x96](../../configs/keypoint/tiny_pose/tinypose_128x96.yml) | 关键点检测 | 支持 | 混合精度 |  | Paddle Inference: C++  |
| [HRNet](https://arxiv.org/abs/1902.09212) | [hrnet_w32_256x192](../../configs/keypoint/hrnet/hrnet_w32_256x192.yml) | 关键点检测 | 支持 | 混合精度 |  | Paddle Inference: C++  |
| HRNet | [dark_hrnet_w32_256x192](../../configs/keypoint/hrnet/dark_hrnet_w32_256x192.yml) | 关键点检测 | 支持 | 混合精度 |  | Paddle Inference: C++  |
| HRNet | [dark_hrnet_w48_256x192](../../configs/keypoint/hrnet/dark_hrnet_w48_256x192.yml) | 关键点检测 | 支持 | 混合精度 |  | Paddle Inference: C++  |
| [HigherHRNet](https://arxiv.org/abs/1908.10357) | [higherhrnet_hrnet_w32_512](../../configs/keypoint/higherhrnet/higherhrnet_hrnet_w32_512.yml) | 关键点检测 | 支持 | 混合精度 |  | Paddle Inference: C++  |
| [FairMot](https://arxiv.org/abs/2004.01888) | [fairmot_dla34_30e_576x320](../../configs/mot/fairmot/fairmot_dla34_30e_576x320.yml) | 目标跟踪 | 支持 | 混合精度 |  | Paddle Inference: C++  |
| FairMot | [fairmot_hrnetv2_w18_dlafpn_30e_576x320](../../configs/mot/fairmot/fairmot_hrnetv2_w18_dlafpn_30e_576x320.yml) | 目标跟踪 | 支持 | 混合精度 |  | Paddle Inference: C++  |
| [JDE](https://arxiv.org/abs/1909.12605) | [jde_darknet53_30e_576x320](../../configs/mot/jde/jde_darknet53_30e_576x320.yml) | 目标跟踪 | 支持 | 混合精度 |  | Paddle Inference: C++  |
