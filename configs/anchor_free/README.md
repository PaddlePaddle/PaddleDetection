# Anchor Free系列模型

## 内容
- [简介](#简介)
- [模型库与基线](#模型库与基线)
- [算法细节](#算法细节)
- [如何贡献代码](#如何贡献代码)

## 简介
目前主流的检测算法大体分为两类： single-stage和two-stage，其中single-stage的经典算法包括SSD, YOLO等，two-stage方法有RCNN系列模型，两大类算法在[PaddleDetection Model Zoo](../../docs/MODEL_ZOO.md)中均有给出，它们的共同特点是先定义一系列密集的，大小不等的anchor区域，再基于这些先验区域进行分类和回归，这种方式极大的受限于anchor自身的设计。随着CornerNet的提出，涌现了多种anchor free方法，PaddleDetection也集成了一系列anchor free算法。

## 模型库与基线
下表中展示了PaddleDetection当前支持的网络结构，具体细节请参考[算法细节](#算法细节)。

|                          | ResNet50  | ResNet50-vd | Hourglass104 |
|:------------------------:|:--------:|:--------------------------:|:------------------------:|
| [CornerNet-Squeeze](#CornerNet-Squeeze)  | x        |                          ✓ | ✓                        |
| [FCOS](#FCOS)  | ✓        |                          x | x                        |


### 模型库

#### COCO数据集上的mAP

| 网络结构 | 骨干网络 | 图片个数/GPU | 预训练模型 | mAP | FPS  | 模型下载 | 配置文件 |
|:------------:|:--------:|:----:|:-------:|:-------:|:---------:|:----------:|:----------:|
| CornerNet-Squeeze    | Hourglass104 | 14  |    无    | 34.5  | 35.5 | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/cornernet_squeeze_hg104.tar)  | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/anchor_free/cornernet_squeeze_hg104.yml) |
| CornerNet-Squeeze    | ResNet50-vd    | 14  |    [faster\_rcnn\_r50\_vd\_fpn\_2x](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_vd_fpn_2x.tar)    | 32.7     | 42.45      | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/cornernet_squeeze_r50_vd_fpn.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/anchor_free/cornernet_squeeze_r50_vd_fpn.yml) |
| CornerNet-Squeeze-dcn    | ResNet50-vd    | 14  |    [faster\_rcnn\_dcn\_r50\_vd\_fpn\_2x](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_dcn_r50_vd_fpn_2x.tar)    | 34.9    | 40.05      | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/cornernet_squeeze_dcn_r50_vd_fpn.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/anchor_free/cornernet_squeeze_dcn_r50_vd_fpn.yml) |
| CornerNet-Squeeze-dcn-mixup-cosine*    | ResNet50-vd    | 14  |    [faster\_rcnn\_dcn\_r50\_vd\_fpn\_2x](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_dcn_r50_vd_fpn_2x.tar)    | 38.2    | 40.05      | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/cornernet_squeeze_dcn_r50_vd_fpn_mixup_cosine.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/anchor_free/cornernet_squeeze_dcn_r50_vd_fpn_mixup_cosine.yml) |
| FCOS    | ResNet50    | 2  |    [ResNet50\_cos\_pretrained](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_cos_pretrained.tar)    | 39.8 | -      | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/fcos_r50_fpn_1x.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/anchor_free/fcos_r50_fpn_1x.yml) |
| FCOS+multiscale_train    | ResNet50    | 2  |    [ResNet50\_cos\_pretrained](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_cos_pretrained.tar)    | 42.0 | -      | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/fcos_r50_fpn_multiscale_2x.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/anchor_free/fcos_r50_fpn_multiscale_2x.yml) |
| FCOS+DCN    | ResNet50    | 2  |    [ResNet50\_cos\_pretrained](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_cos_pretrained.tar)    | 44.4 | -      | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/fcos_dcn_r50_fpn_1x.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/anchor_free/fcos_dcn_r50_fpn_1x.yml) |

**注意:**

- 模型FPS在Tesla V100单卡环境中通过tools/eval.py进行测试
- CornerNet-Squeeze要求使用PaddlePaddle1.8及以上版本或适当的develop版本
- CornerNet-Squeeze中使用ResNet结构的骨干网络时，加入了FPN结构，骨干网络的输出feature map采用FPN中的P3层输出。
- \*CornerNet-Squeeze-dcn-mixup-cosine是基于原版CornerNet-Squeeze优化效果最好的模型，在ResNet的骨干网络基础上增加mixup预处理和使用cosine_decay
- FCOS使用GIoU loss、用location分支预测centerness、左上右下角点偏移量归一化和ground truth中心匹配策略
- Cornernet-Squeeze模型依赖corner_pooling op，该op在```ppdet/ext_op```中编译得到，具体编译方式请参考[自定义OP的编译过程](../../ppdet/ext_op/README.md)

## 算法细节

### CornerNet-Squeeze

**简介:** [CornerNet-Squeeze](https://arxiv.org/abs/1904.08900) 在[Cornernet](https://arxiv.org/abs/1808.01244)基础上进行改进，预测目标框的左上角和右下角的位置，同时参考SqueezeNet和MobileNet的特点，优化了CornerNet骨干网络Hourglass-104，大幅提升了模型预测速度，相较于原版[YOLO-v3](https://arxiv.org/abs/1804.02767)，在训练精度和推理速度上都具备一定优势。

**特点:**  

- 使用corner_pooling获取候选框左上角和右下角的位置
- 替换Hourglass-104中的residual block为SqueezeNet中的fire-module
- 替换第二层3x3卷积为3x3深度可分离卷积


### FCOS

**简介:** [FCOS](https://arxiv.org/abs/1904.01355)是一种密集预测的anchor-free检测算法，使用RetinaNet的骨架，直接在feature map上回归目标物体的长宽，并预测物体的类别以及centerness（feature map上像素点离物体中心的偏移程度），centerness最终会作为权重来调整物体得分。

**特点:**  

- 利用FPN结构在不同层预测不同scale的物体框，避免了同一feature map像素点处有多个物体框重叠的情况
- 通过center-ness单层分支预测当前点是否是目标中心，消除低质量误检


## 如何贡献代码
我们非常欢迎您可以为PaddleDetection中的Anchor Free检测模型提供代码，您可以提交PR供我们review；也十分感谢您的反馈，可以提交相应issue，我们会及时解答。
