# Focal Loss for Dense Object Detection

## Introduction

We reproduce RetinaNet proposed in paper Focal Loss for Dense Object Detection.

## Model Zoo

| Backbone     | Model     | mstrain | imgs/GPU | lr schedule | FPS | Box AP | download                                                                                                                                                                            | config                                |
| ------------ | --------- | ------- | -------- | ----------- | --- | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- |
| ResNet50-FPN | RetinaNet | Yes     | 4        | 1x          | --- | 37.5   | [model](https://bj.bcebos.com/v1/paddledet/models/retinanet_r50_fpn_mstrain_1x_coco.pdparams)\|[log](https://bj.bcebos.com/v1/paddledet/logs/retinanet_r50_fpn_mstrain_1x_coco.log) | retinanet_r50_fpn_mstrain_1x_coco.yml |

**Notes:**

- All above models are trained on COCO train2017 with 4 GPUs and evaludated on val2017. Box AP=`mAP(IoU=0.5:0.95)`.

- Config `configs/retinanet/retinanet_r50_fpn_1x_coco.yml` is for 8 GPUs and `configs/retinanet/retinanet_r50_fpn_mstrain_1x_coco.yml` is for 4 GPUs (mind the difference of train batch size).

## Citation

```latex
@inproceedings{lin2017focal,
  title={Focal loss for dense object detection},
  author={Lin, Tsung-Yi and Goyal, Priya and Girshick, Ross and He, Kaiming and Doll{\'a}r, Piotr},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  year={2017}
}
```
