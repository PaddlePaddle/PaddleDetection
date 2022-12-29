# Co-tuning for Transfer Learning <br />Supervised Contrastive Learning

## Data preparation
以[Kaggle数据集](https://www.kaggle.com/andrewmvd/road-sign-detection) 比赛数据为例，说明如何准备自定义数据。
Kaggle上的 [road-sign-detection](https://www.kaggle.com/andrewmvd/road-sign-detection) 比赛数据包含877张图像，数据类别4类：crosswalk，speedlimit，stop，trafficlight。
可从Kaggle上下载，也可以从[下载链接](https://fsdet-dataset.bj.bcebos.com/roadsign_coco.tar.gz) 下载。
分别从原始数据集中每类选取相同样本（例如：10shots即每类都有十个训练样本）训练即可。<br />
工业数据集使用PKU-Market-PCB，该数据集用于印刷电路板（PCB）的瑕疵检测，提供了6种常见的PCB缺陷[下载链接](https://fsdet-dataset.bj.bcebos.com/pcb.tar.gz)


## Model Zoo
| 骨架网络             | 网络类型       | 每张GPU图片个数 | 每类样本个数 | Box AP |                           下载                          | 配置文件 |
| :------------------- | :------------- | :-----: | :------------: | :-----: | :-----------------------------------------------------: | :-----: |
| ResNet50-vd             | Faster         |    1    |     10     |  60.1  |  [下载链接](https://bj.bcebos.com/v1/paddledet/models/faster_rcnn_r50_vd_fpn_1x_coco.pdparams) | [配置文件](./faster_rcnn_r50_vd_fpn_1x_coco_cotuning_roadsign.yml) |
| PPYOLOE_crn_s             | PPYOLOE         |    1    |     30    |  17.8  | [下载链接](https://bj.bcebos.com/v1/paddledet/models/ppyoloe_plus_crn_s_80e_contrast_pcb.pdparams) |[配置文件](./ppyoloe_plus_crn_s_80e_contrast_pcb.yml) |

## Compare-cotuning
| 骨架网络             | 网络类型       | 每张GPU图片个数 |每类样本个数 | Cotuning |  Box AP  |
| :------------------- | :------------- | :-----: | :-----: | :------------: | :-----: |
| ResNet50-vd             | Faster         |    1    |     10     |  False  |  56.7  |
| ResNet50-vd             | Faster         |    1    |     10     |  True  |  60.1 |

## Compare-contrast
| 骨架网络             | 网络类型       | 每张GPU图片个数 | 每类样本个数 | Contrast |  Box AP  |
| :------------------- | :------------- | :-----: | :-----: | :------------: | :-----: |
| PPYOLOE_crn_s             | PPYOLOE         |    1    |     30    |  False  |  15.4  |
| PPYOLOE_crn_s             | PPYOLOE         |    1    |     30     |  True  |  17.8 |

## Training & Evaluation & Inference
### 1、Training

```
# -c 参数表示指定使用哪个配置文件
# --eval 参数表示边训练边评估，训练过程中会保存验证效果最佳的checkpoint

python tools/train.py -c configs/few-shot/faster_rcnn_r50_vd_fpn_1x_coco_cotuning_roadsign.yml --eval
```
### 2、Evaluation
```
# -c 参数表示指定使用哪个配置文件
# -o 参数表示指定配置文件中的全局变量（覆盖配置文件中的设置）

python tools/eval.py -c configs/few-shot/faster_rcnn_r50_vd_fpn_1x_coco_cotuning_roadsign.yml  \ 
                     -o weights=output/faster_rcnn_r50_vd_fpn_1x_coco_cotuning_roadsign/best_model
```


### 3、Inference
```
# -c 参数表示指定使用哪个配置文件
# --infer_img 参数指定预测图像路径

python tools/infer.py -c configs/few-shot/faster_rcnn_r50_vd_fpn_1x_coco_cotuning_roadsign.yml  \
                       --infer_img=demo/road554.png
```

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