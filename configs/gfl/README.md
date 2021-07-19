# Generalized Focal Loss Model(GFL)

## Introduction

[Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection](https://arxiv.org/abs/2006.04388) and [Generalized Focal Loss V2](https://arxiv.org/pdf/2011.12885.pdf)



## Model Zoo

| Backbone        | Model      | images/GPU | lr schedule |FPS | Box AP |                           download                          | config |
| :-------------- | :------------- | :-----: | :-----: | :------------: | :-----: | :-----------------------------------------------------: | :-----: |
| ResNet50-FPN    | GFL           |    2    |   1x      |     ----     |  40.1  | [download](https://paddledet.bj.bcebos.com/models/gfl_r50_fpn_1x_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/gfl/gfl_r50_fpn_1x_coco.yml) |
| ResNet50-FPN    | GFLv2       |    2    |   1x      |     ----     |  40.4  | [download](https://paddledet.bj.bcebos.com/models/gflv2_r50_fpn_1x_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/gfl/gflv2_r50_fpn_1x_coco.yml) |


**Notes:**

- GFL is trained on COCO train2017 dataset and evaluated on val2017 results of `mAP(IoU=0.5:0.95)`.

## Citations
```
@article{li2020generalized,
  title={Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection},
  author={Li, Xiang and Wang, Wenhai and Wu, Lijun and Chen, Shuo and Hu, Xiaolin and Li, Jun and Tang, Jinhui and Yang, Jian},
  journal={arXiv preprint arXiv:2006.04388},
  year={2020}
}

@article{li2020gflv2,
  title={Generalized Focal Loss V2: Learning Reliable Localization Quality Estimation for Dense Object Detection},
  author={Li, Xiang and Wang, Wenhai and Hu, Xiaolin and Li, Jun and Tang, Jinhui and Yang, Jian},
  journal={arXiv preprint arXiv:2011.12885},
  year={2020}
}

```
