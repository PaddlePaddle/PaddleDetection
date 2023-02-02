# RetinaNet (Focal Loss for Dense Object Detection)

## Model Zoo

| Backbone     | Model     | imgs/GPU | lr schedule | FPS | Box AP | download   | config      |
| ------------ | --------- | -------- | ----------- | --- | ------ | ---------- | ----------- |
| ResNet50-FPN | RetinaNet | 2        | 1x          | --- | 37.5   | [model](https://bj.bcebos.com/v1/paddledet/models/retinanet_r50_fpn_1x_coco.pdparams) | [config](./retinanet_r50_fpn_1x_coco.yml) |
| ResNet50-FPN | RetinaNet | 2        | 2x          | --- | 39.1   | [model](https://bj.bcebos.com/v1/paddledet/models/retinanet_r50_fpn_2x_coco.pdparams) | [config](./retinanet_r50_fpn_2x_coco.yml) |
| ResNet101-FPN| RetinaNet | 2        | 2x          | --- | 40.6   | [model](https://paddledet.bj.bcebos.com/models/retinanet_r101_fpn_2x_coco.pdparams) | [config](./retinanet_r101_fpn_2x_coco.yml)  |
| ResNet50-FPN | RetinaNet + [FGD](../slim/distill/README.md) | 2        | 2x          | --- | 40.8    | [model](https://bj.bcebos.com/v1/paddledet/models/retinanet_r101_distill_r50_2x_coco.pdparams) | [config](./retinanet_r50_fpn_2x_coco.yml)/[slim_config](../slim/distill/retinanet_resnet101_coco_distill.yml) |


**Notes:**

- The ResNet50-FPN are trained on COCO train2017 with 8 GPUs. Both ResNet101-FPN and ResNet50-FPN with [FGD](../slim/distill/README.md) are trained on COCO train2017 with 4 GPUs.
- All above models are evaluated on val2017. Box AP=`mAP(IoU=0.5:0.95)`.


## Citation

```latex
@inproceedings{lin2017focal,
  title={Focal loss for dense object detection},
  author={Lin, Tsung-Yi and Goyal, Priya and Girshick, Ross and He, Kaiming and Doll{\'a}r, Piotr},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  year={2017}
}
```
