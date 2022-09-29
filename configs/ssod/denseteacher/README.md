简体中文 | [English](README_en.md)

# DenseTeacher (DenseTeacher: Dense Pseudo-Label for Semi-supervised Object Detection)

## 模型库

|      模型       |   基础检测器             |  Supervision   |  mAP<sup>val<br>0.5:0.95 |  模型下载  |   配置文件   |
| :------------: | :---------------------: | :-----------: | :---------------: |:-----------: | :---------------: |
| DenseTeacher   |   [fcos_r50_fpn_1x_coco](../fcos/fcos_r50_fpn_1x_coco.yml)  |      1%       |       -        | [download]() | [config](denseteacher/dt_semi_001_fcos_r50_fpn_1x_coco.yml) |
| DenseTeacher   |   [fcos_r50_fpn_1x_coco](../fcos/fcos_r50_fpn_1x_coco.yml)  |      5%       |       -        | [download]() | [config](denseteacher/dt_semi_005_fcos_r50_fpn_1x_coco.yml) |
| DenseTeacher   |   [fcos_r50_fpn_1x_coco](../fcos/fcos_r50_fpn_1x_coco.yml)  |      10%      |       -        | [download]() | [config](denseteacher/dt_semi_010_fcos_r50_fpn_1x_coco.yml) |
| DenseTeacher   |   [fcos_r50_fpn_1x_coco](../fcos/fcos_r50_fpn_1x_coco.yml)  |      full     |       -        | [download]() | [config](denseteacher/dt_semi_full_fcos_r50_fpn_1x_coco.yml) |

**注意:**
- 若训练**纯监督数据**的模型，请参照**基础检测器的配置文件**，只需**修改对应数据集标注路径**即可；
- 半监督检测的配置和使用请参照[文档](../README.md)


## 引用

```

```
