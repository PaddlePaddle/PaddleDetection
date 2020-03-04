English | [简体中文](TRANSFER_LEARNING_cn.md)

# Transfer Learning

Transfer learning aims at learning new knowledge from existing knowledge. For example, take pretrained model from ImageNet to initialize detection models, or take pretrained model from COCO dataset to initialize train detection models in PascalVOC dataset.

In transfer learning, if different dataset and the number of classes is used, the dimensional inconsistency will causes in loading parameters related to the number of classes; On the other hand, if more complicated model is used, need to motify the open-source model construction and selective load parameters. Thus, PaddleDetection should designate parameter fields and ignore loading the parameters which match the fields.

## Transfer Learning in PaddleDetection

In transfer learning, it's needed to load pretrained model selectively. The following two methods can be used:

1. Set `finetune_exclude_pretrained_params` or `finetune_exclude_mismatch_shape` in YAML configuration files. Please refer to [configure file](https://github.com/PaddlePaddle/PaddleDetection/blob/master/configs/yolov3_mobilenet_v1_fruit.yml#L15)
2. Set `finetune_exclude_pretrained_params` or `finetune_exclude_mismatch_shape` in command line. For example:

```python
export PYTHONPATH=$PYTHONPATH:.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -u tools/train.py -c configs/faster_rcnn_r50_1x.yml \
                        -o pretrain_weights=https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_1x.tar \
                           finetune_exclude_pretrained_params=['cls_score','bbox_pred'] \
                           finetune_exclude_mismatch_shape=True
```

* Note:

1. The path in pretrain\_weights is the open-source model link of faster RCNN from COCO dataset. For full models link, please refer to [MODEL_ZOO](../MODEL_ZOO.md)
2. If finetune\_exclude\_mismatch\_shape is set to True, then the parameters from model and pretrain_weights will be matched automatically and the parameters which are mismatched will be ignored. This attr has higher priority than finetune\_exclude\_pretrained\_params.
3. The parameter fields are set in finetune\_exclude\_pretrained\_params. If the name of parameter matches field (wildcard matching), the parameter will be ignored in loading.

If users want to fine-tune by own dataet, and remain the model construction, need to ignore the parameters related to the number of classes. PaddleDetection lists ignored parameter fields corresponding to different model type. The table is shown below: </br>

|      model type    |         ignored parameter fields          |
| :----------------: | :---------------------------------------: |
|     Faster RCNN    |          cls\_score, bbox\_pred           |
|     Cascade RCNN   |          cls\_score, bbox\_pred           |
|       Mask RCNN    | cls\_score, bbox\_pred, mask\_fcn\_logits |
|  Cascade-Mask RCNN | cls\_score, bbox\_pred, mask\_fcn\_logits |
|      RetinaNet     |           retnet\_cls\_pred\_fpn          |
|        SSD         |                ^conv2d\_                  |
|       YOLOv3       |              yolo\_output                 |
