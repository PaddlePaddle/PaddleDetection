# YOLO v4

## 内容
- [简介](#简介)
- [模型库与基线](#模型库与基线)
- [未来工作](#未来工作)
- [如何贡献代码](#如何贡献代码)

## 简介

[YOLO v4](https://arxiv.org/abs/2004.10934)的Paddle实现版本

目前PaddleDetection中转换了[darknet](https://github.com/AlexeyAB/darknet)中YOLO v4的权重，可以直接对图片进行预测，在[test-dev2019](http://cocodataset.org/#detection-2019)中精度为43.5%。另外，PaddleDetection支持VOC数据集上finetune，精度达到86.0%

PaddleDetection支持YOLO v4的多个模块：

- mish激活函数
- PAN模块
- SPP模块
- ciou loss
- label_smooth

## 模型库
下表中展示了PaddleDetection当前支持的网络结构。

|                          | GPU个数 | 测试集  | 骨干网络 |  精度  | 模型下载 |  配置文件  |
|:------------------------:|:-------:|:------:|:--------------------------:|:------------------------:| :---------:| :-----: |
| YOLO v4  | - |test-dev2019        |     CSPDarkNet53 |  43.5 |[下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov4_cspdarknet.pdparams) |  [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/yolo/yolov4_cspdarknet.yml)                   |
| YOLO v4 VOC  | 2 | VOC2007        |     CSPDarkNet53 |  -  |   [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov4_cspdarknet_voc.pdparams) |  [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/yolo/yolov4_cspdarknet_voc.yml)              |

**注意:**

- YOLO v4模型仅支持coco test集评估和图片预测，评估时会将预测结果保存在json文件中，请将结果提交至[cocodataset](http://cocodataset.org/#detection-2019)上查看最终精度指标。
- coco测试集使用test2017，下载请参考[coco2017](http://cocodataset.org/#download)


## 未来工作

1. mish激活函数优化
2. mosaic数据预处理实现
3. scale\_x\_y为yolo_box中decode时对box的位置进行微调，该功能将在Paddle2.0版本中实现


## 如何贡献代码
我们非常欢迎您可以为PaddleDetection提供代码，您可以提交PR供我们review；也十分感谢您的反馈，可以提交相应issue，我们会及时解答。
