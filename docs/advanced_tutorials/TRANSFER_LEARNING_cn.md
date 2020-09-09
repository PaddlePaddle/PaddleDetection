[English](TRANSFER_LEARNING.md) | 简体中文

# 迁移学习教程

迁移学习为利用已有知识，对新知识进行学习。例如利用ImageNet分类预训练模型做初始化来训练检测模型，利用在COCO数据集上的检测模型做初始化来训练基于PascalVOC数据集的检测模型。


### 选择数据

迁移学习需要使用自己的数据集，目前已支持COCO和VOC的数据标注格式，在```tools/x2coco.py```中给出了voc、labelme和cityscape标注格式转换为COCO格式的脚本，具体使用方式可以参考[自定义数据源](READER.md)。数据准备完成后，在配置文件中配置数据路径，对应修改reader中的路径参数即可。

1. COCO数据集需要修改COCODataSet中的参数，以[yolov3\_darknet.yml](https://github.com/PaddlePaddle/PaddleDetection/blob/master/configs/yolov3_darknet.yml#L66)为例，修改yolov3\_reader中的配置：

```yml
  dataset:
    !COCODataSet
      dataset_dir: custom_data/coco # 自定义数据目录
      image_dir: train2017 # 自定义训练集目录，该目录在dataset_dir中
      anno_path: annotations/instances_train2017.json # 自定义数据标注路径，该目录在dataset_dir中  
      with_background: false
```

2. VOC数据集需要修改VOCDataSet中的参数，以[yolov3\_darknet\_voc.yml](https://github.com/PaddlePaddle/PaddleDetection/blob/master/configs/yolov3_darknet_voc.yml#L67)为例：

```yml
  dataset:
    !VOCDataSet
    dataset_dir: custom_data/voc # 自定义数据集目录
    anno_path: trainval.txt # 自定义数据标注路径，该目录在dataset_dir中
    use_default_label: true
    with_background: false

```


### 加载预训练模型


在进行迁移学习时，由于会使用不同的数据集，数据类别数与COCO/VOC数据类别不同，导致在加载开源模型(如COCO预训练模型)时，与类别数相关的权重（例如分类模块的fc层）会出现维度不匹配的问题；另外，如果需要结构更加复杂的模型，需要对已有开源模型结构进行调整，对应权重也需要选择性加载。因此，需要在加载模型时不加载不能匹配的权重。


在迁移学习中，对预训练模型进行选择性加载，支持如下两种迁移学习方式：

#### 直接加载预训练权重（**推荐方式**）

模型中和预训练模型中对应参数形状不同的参数将自动被忽略，例如：

```python
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -u tools/train.py -c configs/faster_rcnn_r50_1x.yml \
                           -o pretrain_weights=https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_1x.tar

```

#### 使用`finetune_exclude_pretrained_params`参数控制忽略参数名

可以显示的指定训练过程中忽略参数的名字，任何参数名均可加入`finetune_exclude_pretrained_params`中，为实现这一目的，可通过如下方式实现：

1. 在 YMAL 配置文件中通过设置`finetune_exclude_pretrained_params`字段。可参考[配置文件](https://github.com/PaddlePaddle/PaddleDetection/blob/master/configs/yolov3_mobilenet_v1_fruit.yml#L15)
2. 在 train.py的启动参数中设置`finetune_exclude_pretrained_params`。例如：

```python
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -u tools/train.py -c configs/faster_rcnn_r50_1x.yml \
                         -o pretrain_weights=https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_1x.tar \
                           finetune_exclude_pretrained_params=['cls_score','bbox_pred'] \
```

* 说明：

1. pretrain\_weights的路径为COCO数据集上开源的faster RCNN模型链接，完整模型链接可参考[MODEL_ZOO](../MODEL_ZOO_cn.md)
2. finetune\_exclude\_pretrained\_params中设置参数字段，如果参数名能够匹配以上参数字段（通配符匹配方式），则在模型加载时忽略该参数。

如果用户需要利用自己的数据进行finetune，模型结构不变，只需要忽略与类别数相关的参数，不同模型类型所对应的忽略参数字段如下表所示：</br>

|      模型类型      |             忽略参数字段                  |
| :----------------: | :---------------------------------------: |
|     Faster RCNN    |          cls\_score, bbox\_pred           |
|     Cascade RCNN   |          cls\_score, bbox\_pred           |
|       Mask RCNN    | cls\_score, bbox\_pred, mask\_fcn\_logits |
|  Cascade-Mask RCNN | cls\_score, bbox\_pred, mask\_fcn\_logits |
|      RetinaNet     |           retnet\_cls\_pred\_fpn          |
|        SSD         |                ^conv2d\_                  |
|       YOLOv3       |              yolo\_output                 |
