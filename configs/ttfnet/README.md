# 1. TTFNet

## 简介

TTFNet是一种用于实时目标检测且对训练时间友好的网络，对CenterNet收敛速度慢的问题进行改进，提出了利用高斯核生成训练样本的新方法，有效的消除了anchor-free head中存在的模糊性。同时简单轻量化的网络结构也易于进行任务扩展。

**特点:**

结构简单，仅需要两个head检测目标位置和大小，并且去除了耗时的后处理操作
训练时间短，基于DarkNet53的骨干网路，V100 8卡仅需要训练2个小时即可达到较好的模型效果

## Model Zoo

| 骨架网络        | 网络类型       | 每张GPU图片个数 | 学习率策略 |推理时间(fps) | Box AP |                           下载                          | 配置文件 |
| :-------------- | :------------- | :-----: | :-----: | :------------: | :-----: | :-----------------------------------------------------: | :-----: |
| DarkNet53    | TTFNet           |    12    |   1x      |     ----     |  33.5  | [下载链接](https://paddledet.bj.bcebos.com/models/ttfnet_darknet53_1x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/ttfnet/ttfnet_darknet53_1x_coco.yml) |





# 2. PAFNet

## 简介

PAFNet（Paddle Anchor Free）是PaddleDetection基于TTFNet的优化模型，精度达到anchor free领域SOTA水平，同时产出移动端轻量级模型PAFNet-Lite

PAFNet系列模型从如下方面优化TTFNet模型：

- [CutMix](https://arxiv.org/abs/1905.04899)
- 更优的骨干网络: ResNet50vd-DCN
- 更大的训练batch size: 8 GPUs，每GPU batch_size=18
- Synchronized Batch Normalization
- [Deformable Convolution](https://arxiv.org/abs/1703.06211)
- [Exponential Moving Average](https://www.investopedia.com/terms/e/ema.asp)
- 更优的预训练模型


## 模型库

| 骨架网络        | 网络类型       | 每张GPU图片个数 | 学习率策略 |推理时间(fps) | Box AP |                           下载                          | 配置文件 |
| :-------------- | :------------- | :-----: | :-----: | :------------: | :-----: | :-----------------------------------------------------: | :-----: |
| ResNet50vd   | PAFNet           |    18    |   10x      |     ----     |  39.8  | [下载链接](https://paddledet.bj.bcebos.com/models/pafnet_10x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/ttfnet/pafnet_10x_coco.yml) |



### PAFNet-Lite

| 骨架网络        | 网络类型       | 每张GPU图片个数 | 学习率策略 | Box AP | 麒麟990延时（ms） | 体积（M）                          | 下载                          | 配置文件 |
| :-------------- | :------------- | :-----: | :-----: | :-----: | :------------: | :-----: | :-----------------------------------------------------: | :-----: |
| MobileNetv3   |  PAFNet-Lite          |    12    |   20x     |     23.9    |  26.00   | 14 | [下载链接](https://paddledet.bj.bcebos.com/models/pafnet_lite_mobilenet_v3_20x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/ttfnet/pafnet_lite_mobilenet_v3_20x_coco.yml) |

**注意：** 由于动态图框架整体升级，PAFNet的PaddleDetection发布的权重模型评估时需要添加--bias字段, 例如

```bash
# 使用PaddleDetection发布的权重
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/ppyolo/pafnet_10x_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/pafnet_10x_coco.pdparams --bias
```

## Citations
```
@article{liu2019training,
  title   = {Training-Time-Friendly Network for Real-Time Object Detection},
  author  = {Zili Liu, Tu Zheng, Guodong Xu, Zheng Yang, Haifeng Liu, Deng Cai},
  journal = {arXiv preprint arXiv:1909.00700},
  year    = {2019}
}
```
