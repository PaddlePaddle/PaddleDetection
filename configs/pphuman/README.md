简体中文 | [English](README.md)

# PP-YOLOE Human 检测模型

PaddleDetection团队提供了针对行人的基于PP-YOLOE的检测模型，用户可以下载模型进行使用。

|    模型   |  数据集  | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 |  下载  | 配置文件 |
|:---------|:-------:|:------:|:------:| :----: | :------:|
|PP-YOLOE-l|   CrowdHuman   |  48.0  |  81.9  | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_36e_crowdhuman.pdparams) | [配置文件](./ppyoloe_crn_l_36e_crowdhuman.yml) |


**注意:**
- PP-YOLOE模型训练过程中使用8 GPUs进行混合精度训练，如果**GPU卡数**或者**batch size**发生了改变，你需要按照公式 **lr<sub>new</sub> = lr<sub>default</sub> * (batch_size<sub>new</sub> * GPU_number<sub>new</sub>) / (batch_size<sub>default</sub> * GPU_number<sub>default</sub>)** 调整学习率。
- 具体使用教程请参考[ppyoloe](../ppyoloe#getting-start)。


## 引用
```
@article{shao2018crowdhuman,
    title={CrowdHuman: A Benchmark for Detecting Human in a Crowd},
    author={Shao, Shuai and Zhao, Zijian and Li, Boxun and Xiao, Tete and Yu, Gang and Zhang, Xiangyu and Sun, Jian},
    journal={arXiv preprint arXiv:1805.00123},
    year={2018}
  }
```
