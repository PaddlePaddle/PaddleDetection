# Supervised Baseline 纯监督模型基线

## COCO数据集模型库

### [FCOS](../../fcos)

|  基础模型          |    监督数据比例   |  mAP<sup>val<br>0.5:0.95 |  模型下载  |   配置文件   |
| :---------------: | :-------------: | :---------------------: |:--------: | :---------: |
| FCOS ResNet50-FPN |        5%       |       21.3        | [download](https://paddledet.bj.bcebos.com/models/ssod/fcos_r50_fpn_2x_coco_sup005.pdparams) | [config](fcos_r50_fpn_2x_coco_sup005.yml) |
| FCOS ResNet50-FPN |        10%      |       26.3        | [download](https://paddledet.bj.bcebos.com/models/ssod/fcos_r50_fpn_2x_coco_sup010.pdparams) | [config](fcos_r50_fpn_2x_coco_sup010.yml) |
| FCOS ResNet50-FPN |        full     |       42.6        | [download](https://paddledet.bj.bcebos.com/models/fcos_r50_fpn_iou_multiscale_2x_coco.pdparams) | [config](../../fcos/fcos_r50_fpn_iou_multiscale_2x_coco.yml) |


### [PP-YOLOE+](../../ppyoloe)

|  基础模型          |    监督数据比例   |  mAP<sup>val<br>0.5:0.95 |  模型下载  |   配置文件   |
| :---------------: | :-------------: | :---------------------: |:--------: | :---------: |
| PP-YOLOE+_s       |        5%      |        32.8       | [download](https://paddledet.bj.bcebos.com/models/ssod/ppyoloe_plus_crn_s_80e_coco_sup005.pdparams) | [config](ppyoloe_plus_crn_s_80e_coco_sup005.yml) |
| PP-YOLOE+_s       |        10%      |       35.3       | [download](https://paddledet.bj.bcebos.com/models/ssod/ppyoloe_plus_crn_s_80e_coco_sup010.pdparams) | [config](ppyoloe_plus_crn_s_80e_coco_sup010.yml) |
| PP-YOLOE+_s       |        full     |       43.7       | [download](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_s_80e_coco.pdparams) | [config](../../ppyoloe/ppyoloe_plus_crn_s_80e_coco.yml) |


### [Faster R-CNN](../../faster_rcnn)

|  基础模型          |    监督数据比例   |  mAP<sup>val<br>0.5:0.95 |  模型下载  |   配置文件   |
| :---------------: | :-------------: | :---------------------: |:--------: | :---------: |
| Faster R-CNN ResNet50-FPN |        10%      |       25.6      | [download](https://paddledet.bj.bcebos.com/models/ssod/faster_rcnn_r50_fpn_2x_coco_sup010.pdparams) | [config](faster_rcnn_r50_fpn_2x_coco_sup010.yml) |
| Faster R-CNN ResNet50-FPN |        full     |       40.0      | [download](https://paddledet.bj.bcebos.com/models/faster_rcnn_r50_fpn_2x_coco.pdparams) | [config](../../configs/faster_rcnn/faster_rcnn_r50_fpn_2x_coco.yml) |


### [RetinaNet](../../retinanet)

|  基础模型          |    监督数据比例   |  mAP<sup>val<br>0.5:0.95 |  模型下载  |   配置文件   |
| :---------------: | :-------------: | :---------------------: |:--------: | :---------: |
| RetinaNet ResNet50-FPN |        10%      |       23.6       | [download](https://paddledet.bj.bcebos.com/models/ssod/retinanet_r50_fpn_2x_coco_sup010.pdparams) | [config](retinanet_r50_fpn_2x_coco_sup010.yml) |
| RetinaNet ResNet50-FPN |        full     |       37.5(1x)       | [download](https://paddledet.bj.bcebos.com/models/retinanet_r50_fpn_1x_coco.pdparams) | [config](../../configs/retinanet/retinanet_r50_fpn_1x_coco.yml) |


**注意:**
 - COCO部分监督数据集请参照 [数据集准备](../README.md) 去下载和准备，各个比例的训练集均为**从train2017中抽取部分百分比的子集**，默认使用`fold`号为1的划分子集，`sup010`表示抽取10%的监督数据训练，`sup005`表示抽取5%，`full`表示全部train2017，验证集均为val2017全量；
 - 抽取部分百分比的监督数据的抽法不同，或使用的`fold`号不同，精度都会因此而有约0.5 mAP之多的差异；
 - PP-YOLOE+ 使用Objects365预训练，其余模型均使用ImageNet预训练；
 - PP-YOLOE+ 训练80 epoch，其余模型均训练24 epoch，；
