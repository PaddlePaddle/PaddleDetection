# YOLOv3

## Model Zoo

### YOLOv3 on COCO

| 骨架网络             | 输入尺寸   | 每张GPU图片个数 | 学习率策略 |推理时间(fps) | Box AP |                           下载                          | 配置文件 |
| :------------------- | :------- | :-----: | :-----: | :------------: | :-----: | :-----------------------------------------------------: | :-----: |
| DarkNet53(paper)  | 608         |    8    |   270e    |     ----     |  33.0  |    -   |    -   |
| DarkNet53(paper)  | 416         |    8    |   270e    |     ----     |  31.0  |    -   |    -   |
| DarkNet53(paper)  | 320         |    8    |   270e    |     ----     |  28.2  |    -   |    -   |
| DarkNet53         | 608         |    8    |   270e    |     ----     |  39.0  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_darknet53_270e_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/yolov3/yolov3_darknet53_270e_coco.yml) |
| DarkNet53         | 416         |    8    |   270e    |     ----     |  37.5  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_darknet53_270e_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/yolov3/yolov3_darknet53_270e_coco.yml) |
| DarkNet53         | 320         |    8    |   270e    |     ----     |  34.6  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_darknet53_270e_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/yolov3/yolov3_darknet53_270e_coco.yml) |
|   ResNet50_vd        | 608        |    8    |   270e    |     ----     |  39.1  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_r50vd_dcn_270e_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/yolov3/yolov3_r50vd_dcn_270e_coco.yml) |
| MobileNet-V1         | 608         |    8    |   270e    |     ----     |  28.8  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v1_270e_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/yolov3/yolov3_mobilenet_v1_270e_coco.yml) |
| MobileNet-V1         | 416         |    8    |   270e    |     ----     |  28.7  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v1_270e_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/yolov3/yolov3_mobilenet_v1_270e_coco.yml) |
| MobileNet-V1         | 320         |    8    |   270e    |     ----     |  26.5  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v1_270e_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/yolov3/yolov3_mobilenet_v1_270e_coco.yml) |
| MobileNet-V3         | 608         |    8    |   270e    |     ----     |  31.4  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v3_large_270e_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/yolov3/yolov3_mobilenet_v3_large_270e_coco.yml) |
| MobileNet-V3         | 416         |    8    |   270e    |     ----     |  29.7  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v3_large_270e_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/yolov3/yolov3_mobilenet_v3_large_270e_coco.yml) |
| MobileNet-V3         | 320         |    8    |   270e    |     ----     |  26.9  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v3_large_270e_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/yolov3/yolov3_mobilenet_v3_large_270e_coco.yml) |

### YOLOv3 on Pasacl VOC

| 骨架网络     | 输入尺寸 | 每张GPU图片个数 | 学习率策略 |推理时间(fps)| Box AP | 下载 | 配置文件 |
| :----------- | :--: | :-----: | :-----: |:------------: |:----: | :-------: | :----: |
| MobileNet-V1 | 608  |    8    |   270e  |      -        |  75.1  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v1_270e_voc.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/yolov3/yolov3_mobilenet_v1_270e_voc.yml) |
| MobileNet-V1 | 416  |    8    |   270e  |      -        |  76.1  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v1_270e_voc.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/yolov3/yolov3_mobilenet_v1_270e_voc.yml) |
| MobileNet-V1 | 320  |    8    |   270e  |      -        |  73.6  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v1_270e_voc.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/yolov3/yolov3_mobilenet_v1_270e_voc.yml) |
| MobileNet-V3 | 608  |    8    |   270e  |      -        |  79.6  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v3_large_270e_voc.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/yolov3/yolov3_mobilenet_v3_large_270e_voc.yml) |
| MobileNet-V3 | 416  |    8    |   270e  |      -        |  78.6  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v3_large_270e_voc.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/yolov3/yolov3_mobilenet_v3_large_270e_voc.yml) |
| MobileNet-V3 | 320  |    8    |   270e  |      -        |  76.4  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v3_large_270e_voc.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/yolov3/yolov3_mobilenet_v3_large_270e_voc.yml) |

**注意：** YOLOv3均使用8GPU训练，训练270个epoch


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
