# Supervised Baseline 纯监督模型基线

## COCO数据集模型库

### [FCOS](../../fcos)

|  基础模型          |    监督数据比例   |    Epochs (Iters)  | mAP<sup>val<br>0.5:0.95 |  模型下载  |   配置文件   |
| :---------------: | :-------------: | :---------------: |:---------------------: |:--------: | :---------: |
| FCOS ResNet50-FPN |        5%       |    24 (8712)      |       21.3        | [download](https://paddledet.bj.bcebos.com/models/fcos_r50_fpn_2x_coco_sup005.pdparams) | [config](fcos_r50_fpn_2x_coco_sup005.yml) |
| FCOS ResNet50-FPN |        10%      |    24 (17424)     |       26.3        | [download](https://paddledet.bj.bcebos.com/models/fcos_r50_fpn_2x_coco_sup010.pdparams) | [config](fcos_r50_fpn_2x_coco_sup010.yml) |
| FCOS ResNet50-FPN |        full     |    24 (175896)    |       42.6        | [download](https://paddledet.bj.bcebos.com/models/fcos_r50_fpn_iou_multiscale_2x_coco.pdparams) | [config](../../fcos/fcos_r50_fpn_iou_multiscale_2x_coco.yml) |

**注意:**
  - 以上模型训练默认使用8 GPUs，总batch_size默认为16，默认初始学习率为0.01。如果改动了总batch_size，请按线性比例相应地调整学习率。


### [PP-YOLOE+](../../ppyoloe)

|  基础模型          |    监督数据比例   |    Epochs (Iters)  |   mAP<sup>val<br>0.5:0.95 |  模型下载  |   配置文件   |
| :---------------: | :-------------: | :---------------: | :---------------------: |:--------: | :---------: |
| PP-YOLOE+_s       |        5%       |    80 (7200)      |       32.8       | [download](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_s_80e_coco_sup005.pdparams) | [config](ppyoloe_plus_crn_s_80e_coco_sup005.yml) |
| PP-YOLOE+_s       |        10%      |    80 (14480)     |       35.3       | [download](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_s_80e_coco_sup010.pdparams) | [config](ppyoloe_plus_crn_s_80e_coco_sup010.yml) |
| PP-YOLOE+_s       |        full     |    80 (146560)    |       43.7       | [download](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_s_80e_coco.pdparams) | [config](../../ppyoloe/ppyoloe_plus_crn_s_80e_coco.yml) |
| PP-YOLOE+_l       |        5%       |    80 (7200)      |       42.9       | [download](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_l_80e_coco_sup005.pdparams) | [config](ppyoloe_plus_crn_l_80e_coco_sup005.yml) |
| PP-YOLOE+_l       |        10%      |    80 (14480)     |       45.7       | [download](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_l_80e_coco_sup010.pdparams) | [config](ppyoloe_plus_crn_l_80e_coco_sup010.yml) |
| PP-YOLOE+_l       |        full     |    80 (146560)    |       49.8       | [download](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_l_80e_coco.pdparams) | [config](../../ppyoloe/ppyoloe_plus_crn_l_80e_coco.yml) |

**注意:**
  - 以上模型训练默认使用8 GPUs，总batch_size默认为64，默认初始学习率为0.001。如果改动了总batch_size，请按线性比例相应地调整学习率。


### [Faster R-CNN](../../faster_rcnn)

|  基础模型          |    监督数据比例   |    Epochs (Iters)  |  mAP<sup>val<br>0.5:0.95 |  模型下载  |   配置文件   |
| :---------------: | :-------------: | :---------------: | :---------------------: |:--------: | :---------: |
| Faster R-CNN ResNet50-FPN |    5%   |    24 (8712)      |       20.7      | [download](https://paddledet.bj.bcebos.com/models/faster_rcnn_r50_fpn_2x_coco_sup005.pdparams) | [config](faster_rcnn_r50_fpn_2x_coco_sup005.yml) |
| Faster R-CNN ResNet50-FPN |    10%  |    24 (17424)     |       25.6      | [download](https://paddledet.bj.bcebos.com/models/faster_rcnn_r50_fpn_2x_coco_sup010.pdparams) | [config](faster_rcnn_r50_fpn_2x_coco_sup010.yml) |
| Faster R-CNN ResNet50-FPN |    full |    24 (175896)    |       40.0      | [download](https://paddledet.bj.bcebos.com/models/faster_rcnn_r50_fpn_2x_coco.pdparams) | [config](../../configs/faster_rcnn/faster_rcnn_r50_fpn_2x_coco.yml) |

**注意:**
  - 以上模型训练默认使用8 GPUs，总batch_size默认为16，默认初始学习率为0.02。如果改动了总batch_size，请按线性比例相应地调整学习率。


### [RetinaNet](../../retinanet)

|  基础模型          |    监督数据比例   |    Epochs (Iters)  |  mAP<sup>val<br>0.5:0.95 |  模型下载  |   配置文件   |
| :---------------: | :-------------: | :---------------: | :---------------------: |:--------: | :---------: |
| RetinaNet ResNet50-FPN |     5%    |    24 (8712)      |       13.9       | [download](https://paddledet.bj.bcebos.com/models/retinanet_r50_fpn_2x_coco_sup005.pdparams) | [config](retinanet_r50_fpn_2x_coco_sup005.yml) |
| RetinaNet ResNet50-FPN |    10%    |    24 (17424)     |       23.6       | [download](https://paddledet.bj.bcebos.com/models/retinanet_r50_fpn_2x_coco_sup010.pdparams) | [config](retinanet_r50_fpn_2x_coco_sup010.yml) |
| RetinaNet ResNet50-FPN |    full   |    24 (175896)    |       39.1       | [download](https://paddledet.bj.bcebos.com/models/retinanet_r50_fpn_2x_coco.pdparams) | [config](../../configs/retinanet/retinanet_r50_fpn_2x_coco.yml) |

**注意:**
  - 以上模型训练默认使用8 GPUs，总batch_size默认为16，默认初始学习率为0.01。如果改动了总batch_size，请按线性比例相应地调整学习率。


### 注意事项
 - COCO部分监督数据集请参照 [数据集准备](../README.md) 去下载和准备，各个比例的训练集均为**从train2017中抽取部分百分比的子集**，默认使用`fold`号为1的划分子集，`sup010`表示抽取10%的监督数据训练，`sup005`表示抽取5%，`full`表示全部train2017，验证集均为val2017全量；
 - 抽取部分百分比的监督数据的抽法不同，或使用的`fold`号不同，精度都会因此而有约0.5 mAP之多的差异；
 - PP-YOLOE+ 使用Objects365预训练，其余模型均使用ImageNet预训练；
 - 线型比例相应调整学习率，参照公式: **lr<sub>new</sub> = lr<sub>default</sub> * (batch_size<sub>new</sub> * GPU_number<sub>new</sub>) / (batch_size<sub>default</sub> * GPU_number<sub>default</sub>)**。


## 使用教程

将以下命令写在一个脚本文件里如```run.sh```，一键运行命令为：```sh run.sh```，也可命令行一句句去运行：

```bash
model_type=semi_det/baseline
job_name=ppyoloe_plus_crn_s_80e_coco_sup010 # 可修改，如 fcos_r50_fpn_2x_coco_sup010

config=configs/${model_type}/${job_name}.yml
log_dir=log_dir/${job_name}
weights=output/${job_name}/model_final.pdparams

# 1.training
# CUDA_VISIBLE_DEVICES=0 python tools/train.py -c ${config}
python -m paddle.distributed.launch --log_dir=${log_dir} --gpus 0,1,2,3,4,5,6,7 tools/train.py -c ${config} --eval --amp

# 2.eval
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c ${config} -o weights=${weights}
```
