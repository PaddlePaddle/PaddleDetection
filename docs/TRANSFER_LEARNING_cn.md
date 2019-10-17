# 迁移学习

迁移学习为利用已有知识，对新知识进行学习。例如利用ImageNet分类预训练模型做初始化来训练检测模型，利用在COCO数据集上的检测模型做初始化来训练基于PascalVOC数据集的检测模型。

在进行迁移学习时，由于会使用不同的数据集，数据类别数与COCO/VOC数据类别不同，导致在加载PaddlePaddle开源模型时，与类别数相关的权重（例如分类模块的fc层）会出现维度不匹配的问题；另外，如果需要结构更加复杂的模型，需要对已有开源模型结构进行调整，对应权重也需要选择性加载。因此，需要检测库能够指定参数字段，在加载模型时不加载匹配的权重。

## PaddleDetection进行迁移学习

在迁移学习中，对预训练模型进行选择性加载，可通过如下两种方式实现：

1. 在 YMAL 配置文件中通过设置`finetune_exclude_pretrained_params`字段。可参考[配置文件](../configs/yolov3_mobilenet_v1_fruit.yml#L15)
2. 在 train.py的启动参数中设置 -o finetune_exclude_pretrained_params。例如：

```python
export PYTHONPATH=$PYTHONPATH:.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -u tools/train.py -c configs/faster_rcnn_r50_1x.yml \
                        -o pretrain_weights=https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_1x.tar \
                           finetune_exclude_pretrained_params=['cls_score','bbox_pred']
```

* 说明：

1. pretrain\_weights的路径为COCO数据集上开源的faster RCNN模型链接，完整模型链接可参考[MODEL_ZOO](MODEL_ZOO_cn.md)
2. finetune\_exclude\_pretrained\_params中设置参数字段，如果参数名能够匹配以上参数字段（通配符匹配方式），则在模型加载时忽略该参数。

如果用户需要利用自己的数据进行finetune，模型结构不变，只需要忽略与类别数相关的参数。PaddleDetection给出了不同模型类型所对应的忽略参数字段。如下表所示：</br>

|      模型类型      |             忽略参数字段                  |
| :----------------: | :---------------------------------------: |
|     Faster RCNN    |          cls\_score, bbox\_pred           |
|     Cascade RCNN   |          cls\_score, bbox\_pred           |
|       Mask RCNN    | cls\_score, bbox\_pred, mask\_fcn\_logits |
|  Cascade-Mask RCNN | cls\_score, bbox\_pred, mask\_fcn\_logits |
|      RetinaNet     |           retnet\_cls\_pred\_fpn          |
|        SSD         |                ^conv2d\_                  |
|       YOLOv3       |              yolo\_output                 |
