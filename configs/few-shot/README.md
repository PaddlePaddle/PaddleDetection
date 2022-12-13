# Co-tuning for Transfer Learning  
Supervised Contrastive Learning

## Data preparation
以[Kaggle数据集](https://www.kaggle.com/andrewmvd/road-sign-detection) 比赛数据为例，说明如何准备自定义数据。
Kaggle上的 [road-sign-detection](https://www.kaggle.com/andrewmvd/road-sign-detection) 比赛数据包含877张图像，数据类别4类：crosswalk，speedlimit，stop，trafficlight。
可从Kaggle上下载，也可以从[下载链接](https://paddlemodels.bj.bcebos.com/object_detection/roadsign_voc.tar) 下载。
分别从原始数据集中每类选取相同样本（例如：10shots即每类都有十个训练样本）训练即可。

## Model Zoo
| 骨架网络             | 网络类型       | 每张GPU图片个数 |推理时间(fps) | Box AP |  配置文件  |
| :------------------- | :------------- | :-----: | :-----: | :------------: | :-----: |
| ResNet50-vd             | Faster         |    1    |     ----     |  60.1  | [配置文件](./faster_rcnn_r50_vd_fpn_1x_coco_cotuning_roadsign.yml) |
| PPYOLOE_crn_s             | PPYOLOE         |    1    |     ----     |  14.4  | [配置文件](./ppyoloe_plus_crn_s_80e_contrast_coco.yml) |

## Compare-cotuning
| 骨架网络             | 网络类型       | 每张GPU图片个数 |推理时间(fps) | Cotuning |  Box AP  |
| :------------------- | :------------- | :-----: | :-----: | :------------: | :-----: |
| ResNet50-vd             | Faster         |    1    |     ----     |  False  |  56.7  |
| ResNet50-vd             | Faster         |    1    |     ----     |  True  |  60.1 |

## Compare-contrast
| 骨架网络             | 网络类型       | 每张GPU图片个数 |推理时间(fps) | Contrast |  Box AP  |
| :------------------- | :------------- | :-----: | :-----: | :------------: | :-----: |
| PPYOLOE_crn_s             | PPYOLOE         |    1    |     ----     |  False  |  12.3  |
| PPYOLOE_crn_s             | PPYOLOE         |    1    |     ----     |  True  |  14.4 |

## Citations
```
@article{you2020co,
  title={Co-tuning for transfer learning},
  author={You, Kaichao and Kou, Zhi and Long, Mingsheng and Wang, Jianmin},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={17236--17246},
  year={2020}
}

@article{khosla2020supervised,
  title={Supervised contrastive learning},
  author={Khosla, Prannay and Teterwak, Piotr and Wang, Chen and Sarna, Aaron and Tian, Yonglong and Isola, Phillip and Maschinot, Aaron and Liu, Ce and Krishnan, Dilip},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={18661--18673},
  year={2020}
}
```