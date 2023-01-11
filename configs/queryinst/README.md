# QueryInst: Instances as Queries

## Introduction

QueryInst is a multi-stage end-to-end system that treats instances of interest as learnable queries, enabling query
based object detectors, e.g., Sparse R-CNN, to have strong instance segmentation performance. The attributes of
instances such as categories, bounding boxes, instance masks, and instance association embeddings are represented by
queries in a unified manner. In QueryInst, a query is shared by both detection and segmentation via dynamic convolutions
and driven by parallelly-supervised multi-stage learning.

## Model Zoo

|   Backbone   | Lr schd | Proposals | MultiScale | RandomCrop | bbox AP | mask AP | Download                                                                                             | Config                                                   |
|:------------:|:-------:|:---------:|:----------:|:----------:|:-------:|:-------:|------------------------------------------------------------------------------------------------------|----------------------------------------------------------|
| ResNet50-FPN |   1x    |    100    |     ×      |     ×      |  42.1   |  37.8   | [model](https://bj.bcebos.com/v1/paddledet/models/queryinst_r50_fpn_1x_pro100_coco.pdparams)         | [config](./queryinst_r50_fpn_1x_pro100_coco.yml)         |
| ResNet50-FPN |   3x    |    300    |     √      |     √      |  47.9   |  42.1   | [model](https://bj.bcebos.com/v1/paddledet/models/queryinst_r50_fpn_ms_crop_3x_pro300_coco.pdparams) | [config](./queryinst_r50_fpn_ms_crop_3x_pro300_coco.yml) |

- COCO val-set evaluation results.
- These configurations are for 4-card training.

Please modify these parameters as appropriate:

```yaml
worker_num: 4
TrainReader:
  use_shared_memory: true
find_unused_parameters: true
```

## Citations

```
@InProceedings{Fang_2021_ICCV,
    author    = {Fang, Yuxin and Yang, Shusheng and Wang, Xinggang and Li, Yu and Fang, Chen and Shan, Ying and Feng, Bin and Liu, Wenyu},
    title     = {Instances As Queries},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {6910-6919}
}
```
