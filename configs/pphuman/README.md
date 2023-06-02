简体中文 | [English](README.md)

# PP-YOLOE Human 检测模型

PaddleDetection团队提供了针对行人的基于PP-YOLOE的检测模型，用户可以下载模型进行使用。PP-Human中使用模型为业务数据集模型，我们同时提供CrowdHuman训练配置，可以使用开源数据进行训练。
其中整理后的COCO格式的CrowdHuman数据集[下载链接](https://bj.bcebos.com/v1/paddledet/data/crowdhuman.zip)，检测类别仅一类 `pedestrian(1)`，原始数据集[下载链接](http://www.crowdhuman.org/download.html)。

相关模型的部署模型均在[PP-Human](../../deploy/pipeline/)项目中使用。

|    模型   |  数据集  | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 |  下载  | 配置文件 |
|:---------|:-------:|:------:|:------:| :----: | :------:|
|PP-YOLOE-s|   CrowdHuman   |  42.5  |  77.9  | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_s_36e_crowdhuman.pdparams) | [配置文件](./ppyoloe_crn_s_36e_crowdhuman.yml) |
|PP-YOLOE-l|   CrowdHuman   |  48.0  |  81.9  | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_36e_crowdhuman.pdparams) | [配置文件](./ppyoloe_crn_l_36e_crowdhuman.yml) |
|PP-YOLOE-s|   业务数据集   |  53.2  |  -  | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_s_36e_pphuman.pdparams) | [配置文件](./ppyoloe_crn_s_36e_pphuman.yml) |
|PP-YOLOE-l|   业务数据集   |  57.8  |  -  | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_36e_pphuman.pdparams) | [配置文件](./ppyoloe_crn_l_36e_pphuman.yml) |
|PP-YOLOE+_t-aux(320)|   业务数据集   |  45.7  |  81.2  | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_t_auxhead_320_60e_pphuman.pdparams) | [配置文件](./ppyoloe_plus_crn_t_auxhead_320_60e_pphuman.yml) |


**注意:**
- PP-YOLOE模型训练过程中使用8 GPUs进行混合精度训练，如果**GPU卡数**或者**batch size**发生了改变，你需要按照公式 **lr<sub>new</sub> = lr<sub>default</sub> * (batch_size<sub>new</sub> * GPU_number<sub>new</sub>) / (batch_size<sub>default</sub> * GPU_number<sub>default</sub>)** 调整学习率。
- 具体使用教程请参考[ppyoloe](../ppyoloe#getting-start)。

# YOLOv3 Human 检测模型

请参考[Human_YOLOv3页面](./pedestrian_yolov3/README_cn.md)

# PP-YOLOE 香烟检测模型
基于PP-YOLOE模型的香烟检测模型，是实现PP-Human中的基于检测的行为识别方案的一环，如何在PP-Human中使用该模型进行吸烟行为识别，可参考[PP-Human行为识别模块](../../deploy/pipeline/docs/tutorials/pphuman_action.md)。该模型检测类别仅包含香烟一类。由于数据来源限制，目前暂无法直接公开训练数据。该模型使用了小目标数据集VisDrone上的权重(参照[visdrone](../visdrone))作为预训练模型，以提升检测效果。

|    模型   |  数据集  | mAP<sup>val<br>0.5:0.95 |  mAP<sup>val<br>0.5 | 下载  | 配置文件 |
|:---------|:-------:|:------:|:------:| :----: | :------:|
| PP-YOLOE-s | 香烟业务数据集 |  39.7 | 79.5 |[下载链接](https://bj.bcebos.com/v1/paddledet/models/pipeline/ppyoloe_crn_s_80e_smoking_visdrone.pdparams) | [配置文件](./ppyoloe_crn_s_80e_smoking_visdrone.yml) |

# PP-HGNet 打电话识别模型
基于PP-HGNet模型实现了打电话行为识别，详细可参考[PP-Human行为识别模块](../../deploy/pipeline/docs/tutorials/pphuman_action.md)。该模型基于[PaddleClas](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/models/PP-HGNet.md#3.3)套件进行训练。此处提供预测模型下载：

|    模型   |  数据集  | Acc | 下载  | 配置文件 |
|:---------|:-------:|:------:| :----: | :------:|
| PP-HGNet | 业务数据集 |  86.85 |[下载链接](https://bj.bcebos.com/v1/paddledet/models/pipeline/PPHGNet_tiny_calling_halfbody.zip) | - |

# HRNet 人体关键点模型
人体关键点模型与ST-GCN模型一起完成[基于骨骼点的行为识别](../../deploy/pipeline/docs/tutorials/pphuman_action.md)方案。关键点模型采用HRNet模型，关于关键点模型相关详细资料可以查看关键点专栏页面[KeyPoint](../keypoint/README.md)。此处提供训练模型下载链接。

|    模型   |  数据集  | AP<sup>val<br>0.5:0.95 | 下载  | 配置文件 |
|:---------|:-------:|:------:| :----: | :------:|
| HRNet | 业务数据集 |  87.1 |[下载链接](https://bj.bcebos.com/v1/paddledet/models/pipeline/dark_hrnet_w32_256x192.pdparams) | [配置文件](./hrnet_w32_256x192.yml) |


# ST-GCN 骨骼点行为识别模型
人体关键点模型与[ST-GCN](https://arxiv.org/abs/1801.07455)模型一起完成[基于骨骼点的行为识别](../../deploy/pipeline/docs/tutorials/pphuman_action.md)方案。
ST-GCN模型基于[PaddleVideo](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/applications/PPHuman)完成训练。
此处提供预测模型下载链接。

|    模型   |  数据集  | AP<sup>val<br>0.5:0.95 | 下载  | 配置文件 |
|:---------|:-------:|:------:| :----: | :------:|
| ST-GCN | 业务数据集 |  87.1 |[下载链接](https://bj.bcebos.com/v1/paddledet/models/pipeline/STGCN.zip) | [配置文件](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/applications/PPHuman/configs/stgcn_pphuman.yaml) |

# PP-TSM 视频分类模型
基于`PP-TSM`模型完成了[基于视频分类的行为识别](../../deploy/pipeline/docs/tutorials/pphuman_action.md)方案。
PP-TSM模型基于[PaddleVideo](https://github.com/PaddlePaddle/PaddleVideo/tree/develop/applications/FightRecognition)完成训练。
此处提供预测模型下载链接。

|    模型   |  数据集  | Acc | 下载  | 配置文件 |
|:---------|:-------:|:------:| :----: | :------:|
| PP-TSM | 组合开源数据集 |  89.06 |[下载链接](https://videotag.bj.bcebos.com/PaddleVideo-release2.3/ppTSM_fight.zip) | [配置文件](https://github.com/PaddlePaddle/PaddleVideo/tree/develop/applications/FightRecognition/pptsm_fight_frames_dense.yaml) |

# PP-HGNet、PP-LCNet 属性识别模型
基于PP-HGNet、PP-LCNet 模型实现了行人属性识别，详细可参考[PP-Human行为识别模块](../../deploy/pipeline/docs/tutorials/pphuman_attribute.md)。该模型基于[PaddleClas](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/models/PP-LCNet.md)套件进行训练。此处提供预测模型下载链接.

|    模型   |  数据集  | mA | 下载  | 配置文件 |
|:---------|:-------:|:------:| :----: | :------:|
| PP-HGNet_small | 业务数据集 |  95.4 |[下载链接](https://bj.bcebos.com/v1/paddledet/models/pipeline/PPHGNet_small_person_attribute_954_infer.zip) | - |
| PP-LCNet | 业务数据集 |  94.5 |[下载链接](https://bj.bcebos.com/v1/paddledet/models/pipeline/PPLCNet_x1_0_person_attribute_945_infer.zip) | [配置文件](https://github.com/PaddlePaddle/PaddleClas/blob/develop/ppcls/configs/PULC/person_attribute/PPLCNet_x1_0.yaml) |


## 引用
```
@article{shao2018crowdhuman,
    title={CrowdHuman: A Benchmark for Detecting Human in a Crowd},
    author={Shao, Shuai and Zhao, Zijian and Li, Boxun and Xiao, Tete and Yu, Gang and Zhang, Xiangyu and Sun, Jian},
    journal={arXiv preprint arXiv:1805.00123},
    year={2018}
  }
```
