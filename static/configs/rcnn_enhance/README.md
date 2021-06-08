## 服务器端实用目标检测方案

### 简介

* 近年来，学术界和工业界广泛关注图像中目标检测任务。基于[PaddleClas](https://github.com/PaddlePaddle/PaddleClas)中SSLD蒸馏方案训练得到的ResNet50_vd预训练模型(ImageNet1k验证集上Top1 Acc为82.39%)，结合PaddleDetection中的丰富算子，飞桨提供了一种面向服务器端实用的目标检测方案PSS-DET(Practical Server Side Detection)。基于COCO2017目标检测数据集，V100单卡预测速度为为61FPS时，COCO mAP可达41.6%；预测速度为20FPS时，COCO mAP可达47.8%。

* 以标准的Faster RCNN ResNet50_vd FPN为例，下表给出了PSS-DET不同的模块的速度与精度收益。

| Trick | Train scale | Test scale |  COCO mAP | Infer speed/FPS |
|- |:-: |:-: | :-: | :-: |
| `baseline` | 640x640 | 640x640 | 36.4% | 43.589 |
| +`test proposal=pre/post topk 500/300` | 640x640 | 640x640 | 36.2% | 52.512 |
| +`fpn channel=64` | 640x640 | 640x640 | 35.1% | 67.450 |
| +`ssld pretrain` | 640x640 | 640x640 | 36.3% | 67.450 |
| +`ciou loss` | 640x640 | 640x640 | 37.1% | 67.450 |
| +`DCNv2` | 640x640 | 640x640 | 39.4% | 60.345 |
| +`3x, multi-scale training` | 640x640 | 640x640 | 41.0% | 60.345 |
| +`auto augment` | 640x640 | 640x640 | 41.4% | 60.345 |
| +`libra sampling` | 640x640 | 640x640 | 41.6% | 60.345 |


基于该实验结论，PaddleDetection结合Cascade RCNN，使用更大的训练与评估尺度(1000x1500)，最终在单卡V100上速度为20FPS，COCO mAP达47.8%。下图给出了目前类似速度的目标检测方法的速度与精度指标。


![pssdet](../../docs/images/pssdet.png)

**注意**
> 这里为了更方便地对比，统一将V100的预测耗时乘以1.2倍，近似转化为Titan V的预测耗时。


### 模型库

| 骨架网络             | 网络类型       | 每张GPU图片个数 | 学习率策略 |推理时间(fps) | Box AP | Mask AP |                           下载                          | 配置文件 |
| :---------------------- | :-------------:  | :-------: | :-----: | :------------: | :----: | :-----: | :-------------: | :-----: |
| ResNet50-vd-FPN-Dcnv2         | Faster     |     2     |   3x    |     61.425     |  41.6  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_dcn_r50_vd_fpn_3x_server_side.tar) |  [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/static/configs/rcnn_enhance/faster_rcnn_dcn_r50_vd_fpn_3x_server_side.yml) |
| ResNet50-vd-FPN-Dcnv2         | Cascade Faster     |     2     |   3x    |     20.001     |  47.8  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/cascade_rcnn_dcn_r50_vd_fpn_3x_server_side.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/static/configs/rcnn_enhance/cascade_rcnn_dcn_r50_vd_fpn_3x_server_side.yml) |
| ResNet101-vd-FPN-Dcnv2         | Cascade Faster     |     2     |   3x    |     19.523     |  49.4  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/cascade_rcnn_dcn_r101_vd_fpn_3x_server_side.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/static/configs/rcnn_enhance/cascade_rcnn_dcn_r101_vd_fpn_3x_server_side.yml) |


**注**：generic文件夹下面的配置文件对应的预训练模型均只支持预测，不支持训练与评估。
