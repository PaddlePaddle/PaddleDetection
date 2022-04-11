# YOLOv3

## Model Zoo

### YOLOv3 on COCO

| 骨架网络             | 输入尺寸   | 每张GPU图片个数 | 学习率策略 |推理时间(fps) | Box AP |                           下载                          | 配置文件 |
| :------------------- | :------- | :-----: | :-----: | :------------: | :-----: | :-----------------------------------------------------: | :-----: |
| DarkNet53(paper)  | 608         |    8    |   270e    |     ----     |  33.0  |    -   |    -   |
| DarkNet53(paper)  | 416         |    8    |   270e    |     ----     |  31.0  |    -   |    -   |
| DarkNet53(paper)  | 320         |    8    |   270e    |     ----     |  28.2  |    -   |    -   |
| DarkNet53         | 608         |    8    |   270e    |     ----     |  39.1  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_darknet53_270e_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/yolov3/yolov3_darknet53_270e_coco.yml) |
| DarkNet53         | 416         |    8    |   270e    |     ----     |  37.7  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_darknet53_270e_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/yolov3/yolov3_darknet53_270e_coco.yml) |
| DarkNet53         | 320         |    8    |   270e    |     ----     |  34.8  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_darknet53_270e_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/yolov3/yolov3_darknet53_270e_coco.yml) |
|   ResNet50_vd        | 608        |    8    |   270e    |     ----     |  40.6  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_r50vd_dcn_270e_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/yolov3/yolov3_r50vd_dcn_270e_coco.yml) |
|   ResNet50_vd        | 416        |    8    |   270e    |     ----     |  38.2  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_r50vd_dcn_270e_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/yolov3/yolov3_r50vd_dcn_270e_coco.yml) |
|   ResNet50_vd        | 320        |    8    |   270e    |     ----     |  35.1  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_r50vd_dcn_270e_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/yolov3/yolov3_r50vd_dcn_270e_coco.yml) |
| ResNet34         | 608         |    8    |   270e    |     ----     |  36.2  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_r34_270e_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/yolov3/yolov3_r34_270e_coco.yml) |
| ResNet34         | 416         |    8    |   270e    |     ----     |  34.3  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_r34_270e_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/yolov3/yolov3_r34_270e_coco.yml) |
| ResNet34         | 320         |    8    |   270e    |     ----     |  31.2  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_r34_270e_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/yolov3/yolov3_r34_270e_coco.yml) |
| MobileNet-V1         | 608         |    8    |   270e    |     ----     |  29.4  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v1_270e_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/yolov3/yolov3_mobilenet_v1_270e_coco.yml) |
| MobileNet-V1         | 416         |    8    |   270e    |     ----     |  29.3  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v1_270e_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/yolov3/yolov3_mobilenet_v1_270e_coco.yml) |
| MobileNet-V1         | 320         |    8    |   270e    |     ----     |  27.2  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v1_270e_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/yolov3/yolov3_mobilenet_v1_270e_coco.yml) |
| MobileNet-V3         | 608         |    8    |   270e    |     ----     |  31.4  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v3_large_270e_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/yolov3/yolov3_mobilenet_v3_large_270e_coco.yml) |
| MobileNet-V3         | 416         |    8    |   270e    |     ----     |  29.6  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v3_large_270e_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/yolov3/yolov3_mobilenet_v3_large_270e_coco.yml) |
| MobileNet-V3         | 320         |    8    |   270e    |     ----     |  27.1  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v3_large_270e_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/yolov3/yolov3_mobilenet_v3_large_270e_coco.yml) |
| MobileNet-V1-SSLD    | 608         |    8    |   270e    |     ----     |  31.0  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v1_ssld_270e_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/yolov3/yolov3_mobilenet_v1_ssld_270e_coco.yml) |
| MobileNet-V1-SSLD    | 416         |    8    |   270e    |     ----     |  30.6  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v1_ssld_270e_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/yolov3/yolov3_mobilenet_v1_ssld_270e_coco.yml) |
| MobileNet-V1-SSLD    | 320         |    8    |   270e    |     ----     |  28.4  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v1_ssld_270e_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/yolov3/yolov3_mobilenet_v1_ssld_270e_coco.yml) |

### YOLOv3 on Pasacl VOC

| 骨架网络     | 输入尺寸 | 每张GPU图片个数 | 学习率策略 |推理时间(fps)| Box AP | 下载 | 配置文件 |
| :----------- | :--: | :-----: | :-----: |:------------: |:----: | :-------: | :----: |
| MobileNet-V1 | 608  |    8    |   270e  |      -        |  75.2  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v1_270e_voc.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/yolov3/yolov3_mobilenet_v1_270e_voc.yml) |
| MobileNet-V1 | 416  |    8    |   270e  |      -        |  76.2  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v1_270e_voc.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/yolov3/yolov3_mobilenet_v1_270e_voc.yml) |
| MobileNet-V1 | 320  |    8    |   270e  |      -        |  74.3  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v1_270e_voc.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/yolov3/yolov3_mobilenet_v1_270e_voc.yml) |
| MobileNet-V3 | 608  |    8    |   270e  |      -        |  79.6  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v3_large_270e_voc.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/yolov3/yolov3_mobilenet_v3_large_270e_voc.yml) |
| MobileNet-V3 | 416  |    8    |   270e  |      -        |  78.6  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v3_large_270e_voc.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/yolov3/yolov3_mobilenet_v3_large_270e_voc.yml) |
| MobileNet-V3 | 320  |    8    |   270e  |      -        |  76.4  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v3_large_270e_voc.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/yolov3/yolov3_mobilenet_v3_large_270e_voc.yml) |
| MobileNet-V1-SSLD | 608  |    8    |   270e  |      -        |  78.3  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v1_ssld_270e_voc.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/yolov3/yolov3_mobilenet_v1_ssld_270e_voc.yml) |
| MobileNet-V1-SSLD | 416  |    8    |   270e  |      -        |  79.6  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v1_ssld_270e_voc.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/yolov3/yolov3_mobilenet_v1_ssld_270e_voc.yml) |
| MobileNet-V1-SSLD | 320  |    8    |   270e  |      -        |  77.3  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v1_ssld_270e_voc.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/yolov3/yolov3_mobilenet_v1_ssld_270e_voc.yml) |
| MobileNet-V3-SSLD | 608  |    8    |   270e  |      -        |  80.4  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v3_large_ssld_270e_voc.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/yolov3/yolov3_mobilenet_v3_large_ssld_270e_voc.yml) |
| MobileNet-V3-SSLD | 416  |    8    |   270e  |      -        |  79.2  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v3_large_ssld_270e_voc.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/yolov3/yolov3_mobilenet_v3_large_ssld_270e_voc.yml) |
| MobileNet-V3-SSLD | 320  |    8    |   270e  |      -        |  77.3  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v3_large_ssld_270e_voc.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/yolov3/yolov3_mobilenet_v3_large_ssld_270e_voc.yml) |

## Citations
```
@misc{redmon2018yolov3,
    title={YOLOv3: An Incremental Improvement},
    author={Joseph Redmon and Ali Farhadi},
    year={2018},
    eprint={1804.02767},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
