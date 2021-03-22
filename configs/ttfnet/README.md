# TTFNet

## 简介

TTFNet是一种用于实时目标检测且对训练时间友好的网络，对CenterNet收敛速度慢的问题进行改进，提出了利用高斯核生成训练样本的新方法，有效的消除了anchor-free head中存在的模糊性。同时简单轻量化的网络结构也易于进行任务扩展。

**特点:**

结构简单，仅需要两个head检测目标位置和大小，并且去除了耗时的后处理操作
训练时间短，基于DarkNet53的骨干网路，V100 8卡仅需要训练2个小时即可达到较好的模型效果

## Model Zoo

| 骨架网络        | 网络类型       | 每张GPU图片个数 | 学习率策略 |推理时间(fps) | Box AP |                           下载                          | 配置文件 |
| :-------------- | :------------- | :-----: | :-----: | :------------: | :-----: | :-----------------------------------------------------: | :-----: |
| DarkNet53    | TTFNet           |    12    |   1x      |     ----     |  33.5  | [下载链接](https://paddledet.bj.bcebos.com/models/ttfnet_darknet53_1x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/ttfnet/ttfnet_darknet53_1x_coco.yml) |

## Citations
```
@article{liu2019training,
  title   = {Training-Time-Friendly Network for Real-Time Object Detection},
  author  = {Zili Liu, Tu Zheng, Guodong Xu, Zheng Yang, Haifeng Liu, Deng Cai},
  journal = {arXiv preprint arXiv:1909.00700},
  year    = {2019}
}
```
