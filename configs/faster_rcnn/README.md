# Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

## Model Zoo

| 骨架网络             | 网络类型       | 每张GPU图片个数 | 学习率策略 |推理时间(fps) | Box AP |                           下载                          | 配置文件 |
| :------------------- | :------------- | :-----: | :-----: | :------------: | :-----: | :-----------------------------------------------------: | :-----: |
| ResNet50             | Faster         |    1    |   1x    |     ----     |  36.7  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_r50_1x_coco.pdparams) | [配置文件](./faster_rcnn_r50_1x_coco.yml) |
| ResNet50-vd          | Faster         |    1    |   1x    |     ----     |  37.6  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_r50_vd_1x_coco.pdparams) | [配置文件](./faster_rcnn_r50_vd_1x_coco.yml) |
| ResNet101            | Faster         |    1    |   1x    |     ----     |  39.0  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_r101_1x_coco.pdparams) | [配置文件](./faster_rcnn_r101_1x_coco.yml) |
| ResNet34-FPN         | Faster         |    1    |   1x    |     ----     |  37.8  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_r34_fpn_1x_coco.pdparams) | [配置文件](./faster_rcnn_r34_fpn_1x_coco.yml) |
| ResNet34-FPN-MultiScaleTest | Faster  |    1    |   1x    |     ----     |  38.2  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_r34_fpn_multiscaletest_1x_coco.pdparams) | [配置文件](./faster_rcnn_r34_fpn_multiscaletest_1x_coco.yml) |
| ResNet34-vd-FPN      | Faster         |    1    |   1x    |     ----     |  38.5  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_r34_vd_fpn_1x_coco.pdparams) | [配置文件](./faster_rcnn_r34_vd_fpn_1x_coco.yml) |
| ResNet50-FPN         | Faster         |    1    |   1x    |     ----     |  38.4  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_r50_fpn_1x_coco.pdparams) | [配置文件](./faster_rcnn_r50_fpn_1x_coco.yml) |
| ResNet50-FPN         | Faster         |    1    |   2x    |     ----     |  40.0  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_r50_fpn_2x_coco.pdparams) | [配置文件](./faster_rcnn_r50_fpn_2x_coco.yml) |
| ResNet50-vd-FPN      | Faster         |    1    |   1x    |     ----     |  39.5  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_r50_vd_fpn_1x_coco.pdparams) | [配置文件](./faster_rcnn_r50_vd_fpn_1x_coco.yml) |
| ResNet50-vd-FPN      | Faster         |    1    |   2x    |     ----     |  40.8  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_r50_vd_fpn_2x_coco.pdparams) | [配置文件](./faster_rcnn_r50_vd_fpn_2x_coco.yml) |
| ResNet101-FPN        | Faster         |    1    |   2x    |     ----     |  41.4  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_r101_fpn_2x_coco.pdparams) | [配置文件](./faster_rcnn_r101_fpn_2x_coco.yml) |
| ResNet101-vd-FPN     | Faster         |    1    |   1x    |     ----     |  42.0  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_r101_vd_fpn_1x_coco.pdparams) | [配置文件](./faster_rcnn_r101_vd_fpn_1x_coco.yml) |
| ResNet101-vd-FPN     | Faster         |    1    |   2x    |     ----     |  43.0  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_r101_vd_fpn_2x_coco.pdparams) | [配置文件](./faster_rcnn_r101_vd_fpn_2x_coco.yml) |
| ResNeXt101-vd-FPN    | Faster         |    1    |   1x    |     ----     |  43.4  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_x101_vd_64x4d_fpn_1x_coco.pdparams) | [配置文件](./faster_rcnn_x101_vd_64x4d_fpn_1x_coco.yml) |
| ResNeXt101-vd-FPN    | Faster         |    1    |   2x    |     ----     |  44.0  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_x101_vd_64x4d_fpn_2x_coco.pdparams) | [配置文件](./faster_rcnn_x101_vd_64x4d_fpn_2x_coco.yml) |
| ResNet50-vd-SSLDv2-FPN | Faster       |    1    |   1x    |     ----     |  41.4  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_r50_vd_fpn_ssld_1x_coco.pdparams) | [配置文件](./faster_rcnn_r50_vd_fpn_ssld_1x_coco.yml) |
| ResNet50-vd-SSLDv2-FPN | Faster       |    1    |   2x    |     ----     |  42.3  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_r50_vd_fpn_ssld_2x_coco.pdparams) | [配置文件](./faster_rcnn_r50_vd_fpn_ssld_2x_coco.yml) |
| Swin-Tiny-FPN | Faster       |    2    |   1x    |     ----     |  42.6  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_swin_tiny_fpn_1x_coco.pdparams) | [配置文件](./faster_rcnn_swin_tiny_fpn_1x_coco.yml) |
| Swin-Tiny-FPN | Faster       |    2    |   2x    |     ----     |  44.8  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_swin_tiny_fpn_2x_coco.pdparams) | [配置文件](./faster_rcnn_swin_tiny_fpn_2x_coco.yml) |
| Swin-Tiny-FPN | Faster       |    2    |   3x    |     ----     |  45.3  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_swin_tiny_fpn_3x_coco.pdparams) | [配置文件](./faster_rcnn_swin_tiny_fpn_3x_coco.yml) |

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
