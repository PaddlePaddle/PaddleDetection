# PP-Human跨镜头跟踪模块

跨镜头跟踪任务，是在单镜头跟踪的基础上，实现不同摄像头中人员的身份匹配关联。在安放、智慧零售等方向有较多的应用。
PP-Human跨镜头跟踪模块主要目的在于提供一套简洁、高效的跨镜跟踪Pipeline，REID模型完全基于开源数据集训练。

## 使用方法

1. 下载模型 [REID模型](https://bj.bcebos.com/v1/paddledet/models/pipeline/reid_model.zip) 并解压到```./output_inference```路径下, MOT模型请参考[mot说明](./mot.md)文件下载。

2. 跨镜头跟踪模式下，要求输入的多个视频放在同一目录下，命令如下：
```python
python3 deploy/pphuman/pipeline.py --config deploy/pphuman/config/infer_cfg.yml --video_dir=[your_video_file_directory] --device=gpu
```

3. 相关配置在`./deploy/pphuman/config/infer_cfg.yml`文件中修改：

```python
python3 deploy/pphuman/pipeline.py
        --config deploy/pphuman/config/infer_cfg.yml
        --video_dir=[your_video_file_directory]
        --device=gpu
        --model_dir reid=reid_best/
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

2. 模型方案为[reid-centroids](https://github.com/mikwieczorek/centroids-reid), Backbone为ResNet50, 主要特色为利用相同id的多个特征提升相似度效果。
本跨镜跟踪中所用REID模型在上述基础上，整合多个开源数据集并压缩模型特征到128维以提升泛化性能。大幅提升了在实际应用中的泛化效果。

### 其他建议
- 提供的REID模型基于开源数据集训练得到，建议加入自有数据，训练更加强有力的REID模型，将非常明显提升跨镜跟踪效果。
- 质量评估部分基于简单逻辑+OpenCV实现，效果有限，如果有条件建议针对性训练质量判断模型。


### 示例效果

- camera 1:
<div width="1080" align="center">
  <img src="./images/c1.gif"/>
</div>

- camera 2:
<div width="1080" align="center">
  <img src="./images/c2.gif"/>
</div>


## 参考文献
```
@article{Wieczorek2021OnTU,
  title={On the Unreasonable Effectiveness of Centroids in Image Retrieval},
  author={Mikolaj Wieczorek and Barbara Rychalska and Jacek Dabrowski},
  journal={ArXiv},
  year={2021},
  volume={abs/2104.13643}
}
```
