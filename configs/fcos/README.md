# FCOS (Fully Convolutional One-Stage Object Detection)

## Model Zoo on COCO

| 骨架网络             | 网络类型       | 每张GPU图片个数 | 学习率策略 |推理时间(fps) | Box AP |                           下载                          | 配置文件 |
| :------------------- | :------------- | :-----: | :-----: | :------------: | :-----: | :-----------------------------------------------------: | :-----: |
| ResNet50-FPN    | FCOS           |    2    |   1x      |     ----     |  39.6  | [download](https://paddledet.bj.bcebos.com/models/fcos_r50_fpn_1x_coco.pdparams) | [config](./fcos_r50_fpn_1x_coco.yml) |
| ResNet50-FPN    | FCOS + iou      |    2    |   1x      |     ----     |  40.0  | [download](https://paddledet.bj.bcebos.com/models/fcos_r50_fpn_iou_1x_coco.pdparams) | [config](./fcos_r50_fpn_iou_1x_coco.yml) |
| ResNet50-FPN    | FCOS + DCN       |    2    |   1x      |     ----     |  44.3  | [download](https://paddledet.bj.bcebos.com/models/fcos_dcn_r50_fpn_1x_coco.pdparams) | [config](./fcos_dcn_r50_fpn_1x_coco.yml) |
| ResNet50-FPN    | FCOS + multiscale_train    |    2    |   2x      |     ----     |  41.8  | [download](https://paddledet.bj.bcebos.com/models/fcos_r50_fpn_multiscale_2x_coco.pdparams) | [config](./fcos_r50_fpn_multiscale_2x_coco.yml) |
| ResNet50-FPN    | FCOS + multiscale_train + iou    |    2    |   2x      |     ----     |  42.6  | [download](https://paddledet.bj.bcebos.com/models/fcos_r50_fpn_iou_multiscale_2x_coco.pdparams) | [config](./fcos_r50_fpn_iou_multiscale_2x_coco.yml) |

**注意:**
  - `+ iou` 表示与原版 FCOS 相比，不使用 `centerness` 而是使用 `iou` 来参与计算loss。
  - 基于 FCOS 的半监督检测方法 `DenseTeaher` 可以参照[DenseTeaher](../semi_det/denseteacher)去使用，结合无标签数据可以进一步提升检测性能。
  - PaddleDetection中默认使用`R50-vb`预训练，如果使用`R50-vd`结合[SSLD](../../../docs/feature_models/SSLD_PRETRAINED_MODEL.md)的预训练模型，可进一步显著提升检测精度，同时backbone部分配置也需要做出相应更改，如：
  ```python
    pretrain_weights: https://paddledet.bj.bcebos.com/models/pretrained/ResNet50_vd_ssld_v2_pretrained.pdparams
    ResNet:
      depth: 50
      variant: d
      norm_type: bn
      freeze_at: 0
      return_idx: [1, 2, 3]
      num_stages: 4
      lr_mult_list: [0.05, 0.05, 0.1, 0.15]
  ```

## Citations
```
@inproceedings{tian2019fcos,
  title   =  {{FCOS}: Fully Convolutional One-Stage Object Detection},
  author  =  {Tian, Zhi and Shen, Chunhua and Chen, Hao and He, Tong},
  booktitle =  {Proc. Int. Conf. Computer Vision (ICCV)},
  year    =  {2019}
}
```
