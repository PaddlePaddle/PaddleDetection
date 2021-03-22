# CascadeCA RCNN
## 简介
CascadeCA RCNN是百度视觉技术部在Google AI Open Images 2019-Object Detction比赛中的最佳单模型，该单模型助力团队在500多参数队伍中取得第二名。Open Images Dataset V5(OIDV5)包含500个类别、173W训练图像和超过1400W个标注边框，是目前已知规模最大的目标检测公开数据集，数据集地址：[https://storage.googleapis.com/openimages/web/index.html](https://storage.googleapis.com/openimages/web/index.html)。团队在比赛中的技术方案报告地址：[https://arxiv.org/pdf/1911.07171.pdf](https://arxiv.org/pdf/1911.07171.pdf)

![](../../images/oidv5_gt.png)

## 方法描述
该模型结合了当前较优的检测方法。具体地，它将ResNet200-vd作为检测模型的骨干网络，其imagenet分类预训练模型可以在[这里](https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/image_classification/README_en.md)下载；结合了CascadeCA RCNN、Feature Pyramid Networks、Non-local、Deformable V2等方法。在这里需要注意的是，标准的CascadeRCNN是只预测2个框（前景和背景，使用得分信息去判断最终前景所属的类别），而该模型对每个类别都单独预测了一个框（Cascade Class Aware）。最终模型框图如下图所示。

![](../../images/oidv5_model_framework.png)


由于OIDV5的类别不均衡现象比较严重，在训练时采用了动态采样的策略去选择样本并进行训练；多尺度训练被用于解决边框面积范围太大的情况；此外，团队使用Libra loss替代Smooth L1 loss，来计算预测框的loss；在预测时，使用SoftNMS方法进行后处理，保证更多的框可以被召回。

Objects365 Dataset和OIDV5有大约189个类别是重复的，因此将两个数据集合并进行训练，用于扩充OIDV5的训练数据，最终该模型与其性能指标如下表所示。更具体的模型训练和融合策略可以见：[OIDV5技术报告](https://arxiv.org/pdf/1911.07171.pdf)。

OIDV5模型训练结果如下。


|        模型结构         | Public/Private Score |                           下载链接                           | 配置文件 |
| :-----------------: | :--------: | :----------------------------------------------------------: | :--------: |
| CascadeCARCNN-FPN-Dcnv2-Nonlocal ResNet200-vd |    0.62690/0.59459    | [模型](https://paddlemodels.bj.bcebos.com/object_detection/oidv5_cascade_rcnn_cls_aware_r200_vd_fpn_dcnv2_nonlocal_softnms.tar) |  [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/oidv5/cascade_rcnn_cls_aware_r200_vd_fpn_dcnv2_nonlocal_softnms.yml) |


此外，为验证模型的性能，PaddleDetection基于该模型结构，也训练了针对COCO2017和Objects365 Dataset的模型，模型和验证集指标如下表。

|        模型结构         | 数据集 |  验证集mAP |                           下载链接                           | 配置文件 |
| :-----------------: | :--------: | :--------: | :----------------------------------------------------------: | :--------: |
| CascadeCARCNN-FPN-Dcnv2-Nonlocal ResNet200-vd | COCO2017 |    51.7%    | [模型](https://paddlemodels.bj.bcebos.com/object_detection/cascade_rcnn_cls_aware_r200_vd_fpn_dcnv2_nonlocal_softnms.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/dcn/cascade_rcnn_cls_aware_r200_vd_fpn_dcnv2_nonlocal_softnms.yml) |
| CascadeCARCNN-FPN-Dcnv2-Nonlocal ResNet200-vd | Objects365 |    34.5%    | [模型](https://paddlemodels.bj.bcebos.com/object_detection/obj365_cascade_rcnn_cls_aware_r200_vd_fpn_dcnv2_nonlocal_softnms.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/obj365/cascade_rcnn_cls_aware_r200_vd_fpn_dcnv2_nonlocal_softnms.yml) |

COCO和Objects365 Dataset数据格式相同，目前只支持预测和评估。

## 使用方法

OIDV5数据集格式与COCO不同，目前仅支持单张图片的预测。OIDV5的模型评估方法可以参考[文档](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/challenge_evaluation.md)

1. 下载模型并解压。

2. 运行预测程序。

```bash
python -u tools/infer.py -c configs/oidv5/cascade_rcnn_cls_aware_r200_vd_fpn_dcnv2_nonlocal_softnms.yml -o weights=./oidv5_cascade_rcnn_cls_aware_r200_vd_fpn_dcnv2_nonlocal_softnms/ --infer_img=demo/000000570688.jpg
```

其中模型所在文件夹需要根据自己放置的位置进行修改。

检测结果图像可以在`output`文件夹中查看。

## 模型检测效果

![](../../images/oidv5_pred.jpg)
