# RetinaNet (Focal Loss for Dense Object Detection)

## Model Zoo

| Backbone     | Model     | imgs/GPU | lr schedule | FPS | Box AP | download   | config      |
| ------------ | --------- | -------- | ----------- | --- | ------ | ---------- | ----------- |
| ResNet50-FPN | RetinaNet | 2        | 1x          | --- | 37.5   | [model](https://bj.bcebos.com/v1/paddledet/models/retinanet_r50_fpn_1x_coco.pdparams) | [config](./retinanet_r50_fpn_1x_coco.yml) |
**Notes:**

- All above models are trained on COCO train2017 with 8 GPUs and evaludated on val2017. Box AP=`mAP(IoU=0.5:0.95)`.


## Citation

```latex
@inproceedings{lin2017focal,
  title={Focal loss for dense object detection},
  author={Lin, Tsung-Yi and Goyal, Priya and Girshick, Ross and He, Kaiming and Doll{\'a}r, Piotr},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  year={2017}
}
```
