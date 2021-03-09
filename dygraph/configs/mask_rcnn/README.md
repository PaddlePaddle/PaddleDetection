# Mask R-CNN

## Model Zoo

| 骨架网络              | 网络类型       | 每张GPU图片个数 | 学习率策略 |推理时间(fps) | Box AP | Mask AP |                           下载                          | 配置文件 |
| :------------------- | :------------| :-----: | :-----: | :------------: | :-----: | :-----: | :-----------------------------------------------------: | :-----: |
| ResNet50             | Mask         |    1    |   1x    |     ----     |  37.4  |    32.8    | [下载链接](https://paddledet.bj.bcebos.com/models/mask_rcnn_r50_1x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/mask_rcnn/mask_rcnn_r50_1x_coco.yml) |
| ResNet50             | Mask         |    1    |   2x    |     ----     |  39.7  |    34.5    | [下载链接](https://paddledet.bj.bcebos.com/models/mask_rcnn_r50_2x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/mask_rcnn/mask_rcnn_r50_2x_coco.yml) |
| ResNet50-FPN         | Mask         |    1    |   1x    |     ----     |  39.2  |    35.6    | [下载链接](https://paddledet.bj.bcebos.com/models/mask_rcnn_r50_fpn_1x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.yml) |
| ResNet50-FPN         | Mask         |    1    |   2x    |     ----     |  40.5  |    36.7    | [下载链接](https://paddledet.bj.bcebos.com/models/mask_rcnn_r50_fpn_2x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/mask_rcnn/mask_rcnn_r50_fpn_2x_coco.yml) |
| ResNet50-vd-FPN         | Mask         |    1    |   1x    |     ----     |  40.3  |    36.4    | [下载链接](https://paddledet.bj.bcebos.com/models/mask_rcnn_r50_vd_fpn_1x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/mask_rcnn/mask_rcnn_r50_vd_fpn_1x_coco.yml) |
| ResNet50-vd-FPN         | Mask         |    1    |   2x    |     ----     |  41.4  |    37.5    | [下载链接](https://paddledet.bj.bcebos.com/models/mask_rcnn_r50_vd_fpn_2x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/mask_rcnn/mask_rcnn_r50_vd_fpn_2x_coco.yml) |
| ResNet101-FPN         | Mask         |    1    |   1x    |     ----     |  40.6  |    36.6    | [下载链接](https://paddledet.bj.bcebos.com/models/mask_rcnn_r101_fpn_1x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/mask_rcnn/mask_rcnn_r101_fpn_1x_coco.yml) |
| ResNet101-vd-FPN         | Mask         |    1    |   1x    |     ----     |  42.4  |    38.1    | [下载链接](https://paddledet.bj.bcebos.com/models/mask_rcnn_r101_vd_fpn_1x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/mask_rcnn/mask_rcnn_r101_vd_fpn_1x_coco.yml) |
| ResNeXt101-vd-FPN        | Mask         |    1    |   1x    |     ----     |  44.0  |    39.5   | [下载链接](https://paddledet.bj.bcebos.com/models/mask_rcnn_x101_vd_64x4d_fpn_1x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/mask_rcnn/mask_rcnn_x101_vd_64x4d_fpn_1x_coco.yml) |
| ResNeXt101-vd-FPN        | Mask         |    1    |   2x    |     ----     |  44.6  |    39.8   | [下载链接](https://paddledet.bj.bcebos.com/models/mask_rcnn_x101_vd_64x4d_fpn_2x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/mask_rcnn/mask_rcnn_x101_vd_64x4d_fpn_2x_coco.yml) |

**注意：** Mask R-CNN模型精度依赖Paddle develop分支修改，精度复现须使用[每日版本](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/Tables.html#whl-dev)或2.0.1版本(将于2021.03发布)，使用Paddle 2.0.0版本会有少量精度损失。

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
