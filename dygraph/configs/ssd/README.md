# SSD: Single Shot MultiBox Detector

## Model Zoo

### SSD on Pascal VOC

| 骨架网络        | 网络类型       | 每张GPU图片个数 | 学习率策略 |推理时间(fps) | Box AP |                           下载                          | 配置文件 |
| :-------------- | :------------- | :-----: | :-----: | :------------: | :-----: | :-----------------------------------------------------: | :-----: |
| VGG             | SSD            |    8    |   240e    |     ----     |  78.2  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/dygraph/ssd_vgg16_300_240e_voc.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/dygraph/configs/ssd_vgg16_300_240e_voc.yml) |

**注意：** SSD使用4GPU训练，训练240个epoch

## Citations
```
@article{Liu_2016,
   title={SSD: Single Shot MultiBox Detector},
   journal={ECCV},
   author={Liu, Wei and Anguelov, Dragomir and Erhan, Dumitru and Szegedy, Christian and Reed, Scott and Fu, Cheng-Yang and Berg, Alexander C.},
   year={2016},
}
```
