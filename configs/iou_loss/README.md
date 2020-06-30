# Improvements of IOU loss

## Introduction

- Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression: [https://arxiv.org/abs/1902.09630](https://arxiv.org/abs/1902.09630)

```
@article{DBLP:journals/corr/abs-1902-09630,
  author    = {Seyed Hamid Rezatofighi and
               Nathan Tsoi and
               JunYoung Gwak and
               Amir Sadeghian and
               Ian D. Reid and
               Silvio Savarese},
  title     = {Generalized Intersection over Union: {A} Metric and {A} Loss for Bounding
               Box Regression},
  journal   = {CoRR},
  volume    = {abs/1902.09630},
  year      = {2019},
  url       = {http://arxiv.org/abs/1902.09630},
  archivePrefix = {arXiv},
  eprint    = {1902.09630},
  timestamp = {Tue, 21 May 2019 18:03:36 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1902-09630},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

- Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression: [https://arxiv.org/abs/1911.08287](https://arxiv.org/abs/1911.08287)

```
@article{Zheng2019DistanceIoULF,
  title={Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression},
  author={Zhaohui Zheng and Ping Wang and Wei Liu and Jinze Li and Rongguang Ye and Dongwei Ren},
  journal={ArXiv},
  year={2019},
  volume={abs/1911.08287}
}
```

## Model Zoo


| Backbone                | Type        |   Loss Type | Loss Weight  | Image/gpu | Lr schd | Inf time (fps) | Box AP | Mask AP |                           Download                           | Configs |
| :---------------------- | :------------- | :---: | :---: | :-------: | :-----: | :------------: | :----: | :-----: | :----------------------------------------------------------: | :---: |
| ResNet50-vd-FPN            | Faster         | GIOU |   10   |    2     |   1x    |     22.94     |  39.4  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_vd_fpn_giou_loss_1x.tar) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/iou_loss/faster_rcnn_r50_vd_fpn_giou_loss_1x.yml) |
| ResNet50-vd-FPN            | Faster         | DIOU |   12   |    2     |   1x    |     22.94     |  39.2  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_vd_fpn_diou_loss_1x.tar) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/iou_loss/faster_rcnn_r50_vd_fpn_diou_loss_1x.yml) |
| ResNet50-vd-FPN            | Faster         | CIOU |   12   |    2     |   1x    |     22.95     |  39.6  |   -   | [model](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_vd_fpn_ciou_loss_1x.tar) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/iou_loss/faster_rcnn_r50_vd_fpn_ciou_loss_1x.yml) |
