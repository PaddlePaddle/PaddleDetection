简体中文 | [English](README.md)

# PP-YOLOE Human 检测模型

PaddleDetection团队提供了针对行人的基于PP-YOLOE的检测模型，用户可以下载模型进行使用。
其中整理后的COCO格式的CrowdHuman数据集[下载链接](https://bj.bcebos.com/v1/paddledet/data/crowdhuman.zip)，检测类别仅一类 `pedestrian(1)`，原始数据集[下载链接](http://www.crowdhuman.org/download.html)。

|    模型   |  数据集  | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 |  下载  | 配置文件 |
|:---------|:-------:|:------:|:------:| :----: | :------:|
|PP-YOLOE-s|   CrowdHuman   |  42.5  |  77.9  | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_s_36e_crowdhuman.pdparams) | [配置文件](./ppyoloe_crn_s_36e_crowdhuman.yml) |
|PP-YOLOE-l|   CrowdHuman   |  48.0  |  81.9  | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_36e_crowdhuman.pdparams) | [配置文件](./ppyoloe_crn_l_36e_crowdhuman.yml) |


**注意:**
- PP-YOLOE模型训练过程中使用8 GPUs进行混合精度训练，如果**GPU卡数**或者**batch size**发生了改变，你需要按照公式 **lr<sub>new</sub> = lr<sub>default</sub> * (batch_size<sub>new</sub> * GPU_number<sub>new</sub>) / (batch_size<sub>default</sub> * GPU_number<sub>default</sub>)** 调整学习率。
- 具体使用教程请参考[ppyoloe](../ppyoloe#getting-start)。


# PP-YOLOE 香烟检测模型
基于PP-YOLOE模型的香烟检测模型，是实现PP-Human中的基于检测的行为识别方案的一环，如何在PP-Human中使用该模型进行吸烟行为识别，可参考[PP-Human行为识别模块](../../deploy/pipeline/docs/tutorials/action.md)。该模型检测类别仅包含香烟一类。由于数据来源限制，目前暂无法直接公开训练数据。该模型使用了小目标数据集Visdrone上的权重作为预训练模型，以提升检测效果。

|    模型   |  数据集  | mAP<sup>val<br>0.5:0.95 |  mAP<sup>val<br>0.5 | 下载  | 配置文件 |
|:---------|:-------:|:------:|:------:| :----: | :------:|

| PP-YOLOE-s | 业务数据集 |  39.7 | 79.5 |[下载链接](https://bj.bcebos.com/v1/paddledet/models/pipeline/ppyoloe_crn_s_80e_smoking_visdrone.pdparams) | [配置文件](./ppyoloe_crn_s_80e_smoking_visdrone.yml) |


## 引用
```
@article{shao2018crowdhuman,
    title={CrowdHuman: A Benchmark for Detecting Human in a Crowd},
    author={Shao, Shuai and Zhao, Zijian and Li, Boxun and Xiao, Tete and Yu, Gang and Zhang, Xiangyu and Sun, Jian},
    journal={arXiv preprint arXiv:1805.00123},
    year={2018}
  }
```
