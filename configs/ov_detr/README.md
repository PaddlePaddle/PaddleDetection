# Open-Vocabulary DETR with Conditional Matching

## Introduction


Open-Vocabulary DETR with Conditional Matching is an object detection model based on DETR++, which, once trained, can detect any object given its class name or an exemplar image.
Code is reproduced by [OV-DETR](https://github.com/yuhangzang/OV-DETR).

## Requirements

- wget
- ftfy
- regex
- paddlepaddle(gpu) >= 2.4.0

## Model Zoo

| Backbone |                     Model                      | Images/GPU | Epochs | Box AP |                       Config                       | init_model  | Log | Download |
|:--------:|:----------------------------------------------:|:----------:|:------:|:------:|:--------------------------------------------------:|-------------|:---:|:--------:|
|   R-50   | Open-Vocabulary DETR with Conditional Matching |     2      |   50   |  待更新   | configs/ov_detr/ov_deformable_detr_r50_1x_coco.yml | [OV-DETR](https://bj.bcebos.com/v1/paddledet/models/pretrained/ov_detr_r50_pre.pdparams) | 待更新 |   待更新    |

**Notes:**

- OV-DETR is trained on open-vocabulary COCO train2017 dataset and evaluated on val2017 results of `mAP(IoU=0.5)`, `seen mAP(IoU=0.5)` and `unseen mAP(IOU=0.5)`.
- OV-DETR uses 8GPU to train 50 epochs.(待更新)

GPU multi-card training
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/deformable_detr/deformable_detr_r50_1x_coco.yml --fleet
```

## Citations
```
@InProceedings{zang2022open,
 author = {Zang, Yuhang and Li, Wei and Zhou, Kaiyang and Huang, Chen an
d Loy, Chen Change},
 title = {Open-Vocabulary DETR with Conditional Matching},
 journal = {European Conference on Computer Vision},
 year = {2022}
}
```

## Acknowledgement

We would like to thanks [CLIP](https://github.com/AgentMaker/Paddle-CLIP) for the open-source projects.
