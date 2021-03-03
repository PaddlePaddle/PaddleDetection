# Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

## Model Zoo

| 骨架网络             | 网络类型       | 每张GPU图片个数 | 学习率策略 |推理时间(fps) | Box AP |                           下载                          | 配置文件 |
| :------------------- | :------------- | :-----: | :-----: | :------------: | :-----: | :-----------------------------------------------------: | :-----: |
| ResNet50             | Faster         |    1    |   1x    |     ----     |  36.7  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_r50_1x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/faster_rcnn/faster_rcnn_r50_1x_coco.yml) |
| ResNet50-vd          | Faster         |    1    |   1x    |     ----     |  37.6  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_r50_vd_1x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/faster_rcnn/faster_rcnn_r50_vd_1x_coco.yml) |
| ResNet101            | Faster         |    1    |   1x    |     ----     |  39.0  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_r101_1x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/faster_rcnn/faster_rcnn_r101_1x_coco.yml) |
| ResNet34-FPN         | Faster         |    1    |   1x    |     ----     |  37.8  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_r34_fpn_1x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/faster_rcnn/faster_rcnn_r34_fpn_1x_coco.yml) |
| ResNet34-vd-FPN      | Faster         |    1    |   1x    |     ----     |  38.5  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_r34_vd_fpn_1x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/faster_rcnn/faster_rcnn_r34_vd_fpn_1x_coco.yml) |
| ResNet50-FPN         | Faster         |    1    |   1x    |     ----     |  38.4  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_r50_fpn_1x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml) |
| ResNet50-FPN         | Faster         |    1    |   2x    |     ----     |  40.0  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_r50_fpn_2x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/faster_rcnn/faster_rcnn_r50_fpn_2x_coco.yml) |
| ResNet50-vd-FPN      | Faster         |    1    |   1x    |     ----     |  39.5  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_r50_vd_fpn_1x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/faster_rcnn/faster_rcnn_r50_vd_fpn_1x_coco.yml) |
| ResNet50-vd-FPN      | Faster         |    1    |   2x    |     ----     |  40.8  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_r50_vd_fpn_2x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/faster_rcnn/faster_rcnn_r50_vd_fpn_2x_coco.yml) |
| ResNet101-FPN        | Faster         |    1    |   2x    |     ----     |  41.4  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_r101_fpn_2x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/faster_rcnn/faster_rcnn_r101_fpn_2x_coco.yml) |
| ResNet101-vd-FPN     | Faster         |    1    |   1x    |     ----     |  42.0  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_r101_vd_fpn_1x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/faster_rcnn/faster_rcnn_r101_vd_fpn_1x_coco.yml) |
| ResNet101-vd-FPN     | Faster         |    1    |   2x    |     ----     |  43.0  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_r101_vd_fpn_2x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/faster_rcnn/faster_rcnn_r101_vd_fpn_2x_coco.yml) |
| ResNeXt101-vd-FPN    | Faster         |    1    |   1x    |     ----     |  43.4  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_x101_vd_64x4d_fpn_1x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/faster_rcnn/faster_rcnn_x101_vd_64x4d_fpn_1x_coco.yml) |
| ResNeXt101-vd-FPN    | Faster         |    1    |   2x    |     ----     |  44.0  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_x101_vd_64x4d_fpn_2x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/faster_rcnn/faster_rcnn_x101_vd_64x4d_fpn_2x_coco.yml) |

**注意：** Faster R-CNN模型精度依赖Paddle develop分支修改，精度复现须使用[每日版本](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/Tables.html#whl-dev)或2.0.1版本(将于2021.03发布)，使用Paddle 2.0.0版本会有少量精度损失。

## Citations
```
@article{Ren_2017,
   title={Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks},
   journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
   publisher={Institute of Electrical and Electronics Engineers (IEEE)},
   author={Ren, Shaoqing and He, Kaiming and Girshick, Ross and Sun, Jian},
   year={2017},
   month={Jun},
}
```
