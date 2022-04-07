# SSD: Single Shot MultiBox Detector

## Model Zoo

### SSD on Pascal VOC

| 骨架网络        | 网络类型       | 每张GPU图片个数 | 学习率策略 |推理时间(fps) | Box AP |                           下载                          | 配置文件 |
| :-------------- | :------------- | :-----: | :-----: | :------------: | :-----: | :-----------------------------------------------------: | :-----: |
| VGG             | SSD            |    8    |   240e    |     ----     |  77.8  | [下载链接](https://paddledet.bj.bcebos.com/models/ssd_vgg16_300_240e_voc.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/ssd/ssd_vgg16_300_240e_voc.yml) |
| MobileNet v1    | SSD            |    32    |   120e    |     ----     |  73.8  | [下载链接](https://paddledet.bj.bcebos.com/models/ssd_mobilenet_v1_300_120e_voc.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/ssd/ssd_mobilenet_v1_300_120e_voc.yml) |

**注意：** SSD-VGG使用4GPU在总batch size为32下训练240个epoch。SSD-MobileNetv1使用2GPU在总batch size为64下训练120周期。

## Citations
```
@article{Liu_2016,
   title={SSD: Single Shot MultiBox Detector},
   journal={ECCV},
   author={Liu, Wei and Anguelov, Dragomir and Erhan, Dumitru and Szegedy, Christian and Reed, Scott and Fu, Cheng-Yang and Berg, Alexander C.},
   year={2016},
}
```
