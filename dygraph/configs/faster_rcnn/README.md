# Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

## Model Zoo

| 骨架网络             | 网络类型       | 每张GPU图片个数 | 学习率策略 |推理时间(fps) | Box AP | Mask AP |                           下载                          | 配置文件 |
| :------------------- | :------------- | :-----: | :-----: | :------------: | :-----: | :-----: | :-----------------------------------------------------: | :-----: |
| ResNet50             | Faster         |    1    |   1x    |     ----     |  35.1  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/dygraph/faster_rcnn_r50_1x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/dygraph/configs/faster_rcnn/faster_rcnn_r50_1x_coco.yml) |
| ResNet50-FPN         | Faster         |    1    |   1x    |     ----     |  37.0  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/dygraph/faster_rcnn_r50_fpn_1x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/dygraph/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml) |
| ResNet34-FPN         | Faster         |    2    |   1x    |     ----     |  36.7  |   -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/dygraph/faster_rcnn_r34_fpn_1x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/dygraph/configs/faster_rcnn/faster_rcnn_r34_fpn_1x_coco.yml) |
| ResNet101            | Faster         |    1    |   1x    |     ----     |  38.3  |   -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/dygraph/faster_rcnn_r101_1x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/dygraph/configs/faster_rcnn/faster_rcnn_r101_1x_coco.yml) |
| ResNet101-FPN        | Faster         |    1    |   1x    |     ----     |  38.7  |   -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/dygraph/faster_rcnn_r101_fpn_1x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/dygraph/configs/faster_rcnn/faster_rcnn_r101_fpn_1x_coco.yml) |

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
