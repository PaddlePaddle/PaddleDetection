[English](pphuman_mtmct_en.md) | 简体中文

# PP-Human跨镜头跟踪模块

跨镜头跟踪任务，是在单镜头跟踪的基础上，实现不同摄像头中人员的身份匹配关联。在安放、智慧零售等方向有较多的应用。
PP-Human跨镜头跟踪模块主要目的在于提供一套简洁、高效的跨镜跟踪Pipeline，REID模型完全基于开源数据集训练。

## 使用方法

1. 下载模型 [行人跟踪](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)和[REID模型](https://bj.bcebos.com/v1/paddledet/models/pipeline/reid_model.zip) 并解压到```./output_inference```路径下，修改配置文件中模型路径。也可简单起见直接用默认配置，自动下载模型。 MOT模型请参考[mot说明](./pphuman_mot.md)文件下载。

2. 跨镜头跟踪模式下，要求输入的多个视频放在同一目录下，同时开启infer_cfg_pphuman.yml 中的REID选择中的enable=True, 命令如下：
```python
python3 deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_pphuman.yml --video_dir=[your_video_file_directory] --device=gpu
```

3. 相关配置在`./deploy/pipeline/config/infer_cfg_pphuman.yml`文件中修改：

```python
python3 deploy/pipeline/pipeline.py
        --config deploy/pipeline/config/infer_cfg_pphuman.yml -o REID.model_dir=reid_best/
        --video_dir=[your_video_file_directory]
        --device=gpu
```

## 方案说明

跨镜头跟踪模块，主要由跨镜头跟踪Pipeline及REID模型两部分组成。
1. 跨镜头跟踪Pipeline

```

单镜头跟踪[id+bbox]
        │
根据bbox截取原图中目标——│
        │            │
    REID模型      质量评估(遮挡、完整度、亮度等)
        │            │
    [feature]        [quality]
        │            │
   datacollector—————│
        │
      特征排序、筛选
        │
 多视频各id相似度计算
        │
  id聚类、重新分配id
```

2. 模型方案为[reid-strong-baseline](https://github.com/michuanhaohao/reid-strong-baseline), Backbone为ResNet50, 主要特色为模型结构简单。
本跨镜跟踪中所用REID模型在上述基础上，整合多个开源数据集并压缩模型特征到128维以提升泛化性能。大幅提升了在实际应用中的泛化效果。

### 其他建议
- 提供的REID模型基于开源数据集训练得到，建议加入自有数据，训练更加强有力的REID模型，将非常明显提升跨镜跟踪效果。
- 质量评估部分基于简单逻辑+OpenCV实现，效果有限，如果有条件建议针对性训练质量判断模型。


### 示例效果

- camera 1:
<div width="600" align="center">
  <img src="https://user-images.githubusercontent.com/22989727/205595795-fd859feb-8218-450f-a109-91c27713a662.gif"/>
</div>

- camera 2:
<div width="600" align="center">
  <img src="https://user-images.githubusercontent.com/22989727/205595826-18ab5f0e-a572-4950-a502-96e6eb904a1e.gif"/>
</div>


## 参考文献
```
@InProceedings{Luo_2019_CVPR_Workshops,
author = {Luo, Hao and Gu, Youzhi and Liao, Xingyu and Lai, Shenqi and Jiang, Wei},
title = {Bag of Tricks and a Strong Baseline for Deep Person Re-Identification},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {June},
year = {2019}
}

@ARTICLE{Luo_2019_Strong_TMM,
author={H. {Luo} and W. {Jiang} and Y. {Gu} and F. {Liu} and X. {Liao} and S. {Lai} and J. {Gu}},
journal={IEEE Transactions on Multimedia},
title={A Strong Baseline and Batch Normalization Neck for Deep Person Re-identification},
year={2019},
pages={1-1},
doi={10.1109/TMM.2019.2958756},
ISSN={1941-0077},
}
```
