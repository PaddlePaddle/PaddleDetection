# YOLO v4 模型

## 内容
- [简介](#简介)
- [模型库与基线](#模型库与基线)
- [未来工作](#未来工作)
- [如何贡献代码](#如何贡献代码)

## 简介

[YOLO v4](https://arxiv.org/abs/2004.10934)的Paddle实现版本，要求使用PaddlePaddle2.0.0及以上版本或适当的develop版本

目前转换了[darknet](https://github.com/AlexeyAB/darknet)中YOLO v4的权重，可以直接对图片进行预测，在[test-dev2019](http://cocodataset.org/#detection-2019)中精度为43.5%。另外，支持VOC数据集上finetune，精度达到85.5%

目前支持YOLO v4的多个模块：

- mish激活函数
- PAN模块
- SPP模块
- ciou loss
- label_smooth
- grid_sensitive

目前支持YOLO系列的Anchor聚类算法
``` bash
python tools/anchor_cluster.py -c ${config} -m ${method} -s ${size}
```
主要参数配置参考下表
|    参数    |    用途    |    默认值    |    备注    |
|:------:|:------:|:------:|:------:|
| -c/--config | 模型的配置文件 | 无默认值 | 必须指定 |
| -n/--n | 聚类的簇数 | 9 | Anchor的数目 |
| -s/--size | 图片的输入尺寸 | None | 若指定，则使用指定的尺寸，如果不指定, 则尝试从配置文件中读取图片尺寸 |
|  -m/--method  |  使用的Anchor聚类方法  |  v2  |  目前只支持yolov2/v5的聚类算法  |
|  -i/--iters  |  kmeans聚类算法的迭代次数  |  1000  | kmeans算法收敛或者达到迭代次数后终止 |
| -gi/--gen_iters |  遗传算法的迭代次数  | 1000 |  该参数只用于yolov5的Anchor聚类算法  |
| -t/--thresh|  Anchor尺度的阈值  | 0.25 | 该参数只用于yolov5的Anchor聚类算法 |

## 模型库
下表中展示了当前支持的网络结构。

|                          | GPU个数 | 测试集  | 骨干网络 |  精度  | 模型下载 |  配置文件  |
|:------------------------:|:-------:|:------:|:--------------------------:|:------------------------:| :---------:| :-----: |
| YOLO v4  | - |test-dev2019        |     CSPDarkNet53 |  43.5 |[下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov4_cspdarknet.pdparams) |  [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/yolov4/yolov4_cspdarknet.yml)                   |
| YOLO v4 VOC  | 2 | VOC2007        |     CSPDarkNet53 |  85.5  |   [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov4_cspdarknet_voc.pdparams) |  [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/yolov4/yolov4_cspdarknet_voc.yml)              |

**注意:**

- 由于原版YOLO v4使用coco trainval2014进行训练，训练样本中包含部分评估样本，若使用val集会导致精度虚高，因此使用coco test集对模型进行评估。
- YOLO v4模型仅支持coco test集评估和图片预测，由于test集不包含目标框的真实标注，评估时会将预测结果保存在json文件中，请将结果提交至[cocodataset](http://cocodataset.org/#detection-2019)上查看最终精度指标。
- coco测试集使用test2017，下载请参考[coco2017](http://cocodataset.org/#download)


## 未来工作

1. mish激活函数优化
2. mosaic数据预处理实现



## 如何贡献代码
我们非常欢迎您可以为PaddleDetection提供代码，您可以提交PR供我们review；也十分感谢您的反馈，可以提交相应issue，我们会及时解答。
