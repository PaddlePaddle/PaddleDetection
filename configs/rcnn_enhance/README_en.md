## Practical Server Side Detection

### Introduction

* In recent years, the object detection task in image has been widely concerned by academia and industry. ResNet50vd pretraining model based on SSLD distillation program training in [PaddleClas](https://github.com/PaddlePaddle/PaddleClas) (Top1 on ImageNet1k verification set) Acc is 82.39%), combined with the rich operator of PaddleDetection, PaddlePaddle provides a practical server side detection scheme PSS-DET(Practical Server Side Detection). Based on COCO2017 object detection dataset, V100 single gpu prediction speed is 61FPS, COCO mAP can reach 41.2%.


### Model library

| Backbone              | Network type | Number of images per GPU | Learning rate strategy | Inferring time(fps) | Box AP | Mask AP |                                      Download                                       |                                                           Configuration File                                                            |
| :-------------------- | :----------: | :----------------------: | :--------------------: | :-----------------: | :----: | :-----: | :---------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------: |
| ResNet50-vd-FPN-Dcnv2 |    Faster    |            2             |           3x           |       61.425        |  41.5  |    -    | [link](https://paddledet.bj.bcebos.com/models/faster_rcnn_enhance_3x_coco.pdparams) | [Configuration File](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/rcnn_enhance/faster_rcnn_enhance_3x_coco.yml) |
