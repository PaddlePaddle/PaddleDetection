# Generalized Focal Loss Model(GFL)

## Introduction

We reproduce the object detection results in the paper [Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection](https://arxiv.org/abs/2006.04388) and [Generalized Focal Loss V2](https://arxiv.org/pdf/2011.12885.pdf). And We use a better performing pre-trained model and ResNet-vd structure to improve mAP.

## Model Zoo

| Backbone        | Model      | batch-size/GPU | lr schedule |FPS | Box AP |                           download                          | config |
| :-------------- | :------------- | :-----: | :-----: | :------------: | :-----: | :-----------------------------------------------------: | :-----: |
| ResNet50    | GFL           |    2    |   1x      |     ----     |  41.0  | [model](https://paddledet.bj.bcebos.com/models/gfl_r50_fpn_1x_coco.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_gfl_r50_fpn_1x_coco.log) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/gfl/gfl_r50_fpn_1x_coco.yml) |
| ResNet50    | GFL + [CWD](../slim/README.md) |    2    |   2x      |     ----     |  44.0  | [model](https://paddledet.bj.bcebos.com/models/gfl_r50_fpn_2x_coco_cwd.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_gfl_r50_fpn_2x_coco_cwd.log) | [config1](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/gfl/gfl_r50_fpn_1x_coco.yml), [config2](../slim/distill/gfl_r101vd_fpn_coco_distill_cwd.yml) |
| ResNet101-vd   | GFL           |    2    |   2x      |     ----     |  46.8  | [model](https://paddledet.bj.bcebos.com/models/gfl_r101vd_fpn_mstrain_2x_coco.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_gfl_r101vd_fpn_mstrain_2x_coco.log) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/gfl/gfl_r101vd_fpn_mstrain_2x_coco.yml) |
| ResNet34-vd    | GFL           |    2    |   1x      |     ----     |  40.8  | [model](https://paddledet.bj.bcebos.com/models/gfl_r34vd_1x_coco.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_gfl_r34vd_1x_coco.log) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/gfl/gfl_r34vd_1x_coco.yml) |
| ResNet18-vd   | GFL           |    2    |   1x      |     ----     |  36.6  | [model](https://paddledet.bj.bcebos.com/models/gfl_r18vd_1x_coco.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_gfl_r18vd_1x_coco.log) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/gfl/gfl_r18vd_1x_coco.yml) |
| ResNet18-vd   | GFL + [LD](../slim/README.md)    |    2    |   1x      |     ----     |  38.2  | [model](https://bj.bcebos.com/v1/paddledet/models/gfl_slim_ld_r18vd_1x_coco.pdparams) &#124; [log](https://bj.bcebos.com/v1/paddledet/logs/train_gfl_slim_ld_r18vd_1x_coco.log) | [config1](./gfl_slim_ld_r18vd_1x_coco.yml), [config2](../slim/distill/gfl_ld_distill.yml) |
| ResNet50    | GFLv2       |    2    |   1x      |     ----     |  41.2  | [model](https://paddledet.bj.bcebos.com/models/gflv2_r50_fpn_1x_coco.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_gflv2_r50_fpn_1x_coco.log) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/gfl/gflv2_r50_fpn_1x_coco.yml) |


**Notes:**

- GFL is trained on COCO train2017 dataset with 8 GPUs and evaluated on val2017 results of `mAP(IoU=0.5:0.95)`.

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
