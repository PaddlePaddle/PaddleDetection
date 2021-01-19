# Mask R-CNN

## Model Zoo

| 骨架网络              | 网络类型       | 每张GPU图片个数 | 学习率策略 |推理时间(fps) | Box AP | Mask AP |                           下载                          | 配置文件 |
| :------------------- | :------------| :-----: | :-----: | :------------: | :-----: | :-----: | :-----------------------------------------------------: | :-----: |
| ResNet50             | Mask         |    1    |   1x    |     ----     |  36.4  |    31.9    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/dygraph/mask_rcnn_r50_1x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/mask_rcnn/mask_rcnn_r50_1x_coco.yml) |
| ResNet50             | Mask         |    1    |   2x    |     ----     |  38.2  |    33.4    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/dygraph/mask_rcnn_r50_2x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/mask_rcnn/mask_rcnn_r50_2x_coco.yml) |
| ResNet50-FPN         | Mask         |    1    |   1x    |     ----     |  38.3  |    34.5    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/dygraph/mask_rcnn_r50_fpn_1x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.yml) |
| ResNet50-FPN         | Mask         |    1    |   2x    |     ----     |  38.7  |    34.7    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/dygraph/mask_rcnn_r50_fpn_2x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/mask_rcnn/mask_rcnn_r50_fpn_2x_coco.yml) |
| ResNet50-vd-FPN      | Mask         |    1    |   2x    |     ----     |  39.8  |    35.4    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/dygraph/mask_rcnn_r50_vd_fpn_2x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/mask_rcnn/mask_rcnn_r50_vd_fpn_2x_coco.yml) |
| ResNet101-FPN        | Mask         |    1    |   1x    |     ----     |  39.5  |    35.2    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/dygraph/mask_rcnn_r101_fpn_1x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/mask_rcnn/mask_rcnn_r101_fpn_1x_coco.yml) |
| ResNet101-vd-FPN     | Mask         |    1    |   1x    |     ----     |  41.4  |    36.8    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/dygraph/mask_rcnn_r101_vd_fpn_1x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/mask_rcnn/mask_rcnn_r101_vd_fpn_1x_coco.yml) |
| ResNeXt101-vd-FPN    | Mask         |    1    |   1x    |     ----     |  42.9  |    37.9    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/dygraph/mask_rcnn_x101_vd_64x4d_fpn_1x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/mask_rcnn/mask_rcnn_x101_vd_64x4d_fpn_1x_coco.yml) |

## Citations
```
@article{He_2017,
   title={Mask R-CNN},
   journal={2017 IEEE International Conference on Computer Vision (ICCV)},
   publisher={IEEE},
   author={He, Kaiming and Gkioxari, Georgia and Dollar, Piotr and Girshick, Ross},
   year={2017},
   month={Oct}
}
```
