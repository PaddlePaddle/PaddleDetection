简体中文 | [English](README.md)

# CenterNet (CenterNet: Objects as Points)

## 内容
- [简介](#简介)
- [模型库](#模型库)
- [引用](#引用)

## 内容

[CenterNet](http://arxiv.org/abs/1904.07850)是Anchor Free检测器，将物体表示为一个目标框中心点。CenterNet使用关键点检测的方式定位中心点并回归物体的其他属性。CenterNet是以中心点为基础的检测方法，是端到端可训练的，并且相较于基于anchor的检测器更加检测高效。

## 模型库

### CenterNet在COCO-val 2017上结果

| 骨干网络       | 输入尺寸 | mAP   |    FPS    | 下载链接 | 配置文件 |
| :--------------| :------- |  :----: | :------: | :----: |:-----: |
| DLA-34(paper)  | 512x512 |  37.4  |     -   |    -   |   -    |
| DLA-34         | 512x512 |  37.6  |     -   | [下载链接](https://bj.bcebos.com/v1/paddledet/models/centernet_dla34_140e_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/centernet/centernet_dla34_140e_coco.yml) |
| ResNet50 + DLAUp  | 512x512 |  38.9  |     -   | [下载链接](https://bj.bcebos.com/v1/paddledet/models/centernet_r50_140e_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/centernet/centernet_r50_140e_coco.yml) |
| GhostNet_x1_3 + DLAUp  | 512x512 | 28.9  |     -   | [下载链接](https://paddledet.bj.bcebos.com/models/centernet_ghostnet_1_3x_140e_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/centernet/centernet_ghostnet_1_3x_140e_coco.yml) |
| PP-LCNet_x1_0 + DLAUp  | 512x512 | 26.9  |     -   | [下载链接](https://paddledet.bj.bcebos.com/models/centernet_lcnet_1x_140e_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/centernet/centernet_lcnet_1x_140e_coco.yml) |
| MobileNetV1_1x + DLAUp  | 512x512 |  28.2  |     -   | [下载链接](https://paddledet.bj.bcebos.com/models/centernet_mbv1_x1_0_140e_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/centernet/centernet_mbv1_1x_140e_coco.yml) |
| MobileNetV3_large_x1_0 + DLAUp  | 512x512 |  27.1  |     -   | [下载链接](https://paddledet.bj.bcebos.com/models/centernet_mbv3_large_x1_0_140e_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/centernet/centernet_mbv3_large_1x_140e_coco.yml) |
| ShuffleNetV2_x1_0 + DLAUp  | 512x512 | 23.8  |     -   | [下载链接](https://paddledet.bj.bcebos.com/models/centernet_shufflenetv2_1x_140e_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/centernet/centernet_shufflenetv2_1x_140e_coco.yml) |

## 引用
```
@article{zhou2019objects,
  title={Objects as points},
  author={Zhou, Xingyi and Wang, Dequan and Kr{\"a}henb{\"u}hl, Philipp},
  journal={arXiv preprint arXiv:1904.07850},
  year={2019}
}
```
