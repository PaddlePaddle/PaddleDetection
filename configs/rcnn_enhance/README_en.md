## Server-end practical target detection scheme

### Introduction

* In recent years, the target detection task in image has been widely concerned by academia and industry. ResNet50VD pretraining model based on SSLD distillation program training in [PaddleClas](https://github.com/PaddlePaddle/PaddleClas) (Top1 on Image Net1k verification set) Acc is 82.39%), combined with the rich operator of PaddleDetection, PaddlePaddle provides a Practical Server Side Detection scheme PSS DET(Practical Server Side Detection). Based on COCO2017 target detection data set, when V100 single card predicted speed is 61FPS, COCO mAP can reach 41.2%.


### Model library

| Skeleton network      | Network type | Number of images per GPU | Learning rate strategy | Inferring time(fps) | Box AP | Mask AP |                                      Download                                       |                                                           Configuration File                                                            |
| :-------------------- | :----------: | :----------------------: | :--------------------: | :-----------------: | :----: | :-----: | :---------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------: |
| ResNet50-vd-FPN-Dcnv2 |    Faster    |            2             |           3x           |       61.425        |  41.5  |    -    | [link](https://paddledet.bj.bcebos.com/models/faster_rcnn_enhance_3x_coco.pdparams) | [Configuration File](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/rcnn_enhance/faster_rcnn_enhance_3x_coco.yml) |
