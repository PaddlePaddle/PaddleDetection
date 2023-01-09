# DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection

## Introduction


[DINO](https://arxiv.org/abs/2203.03605) is an object detection model based on DETR. We reproduced the model of the paper.


## Model Zoo

| Backbone |      Model      | Epochs | Box AP |                 Config                  |                                     Download                                     |
|:------:|:---------------:|:------:|:------:|:---------------------------------------:|:--------------------------------------------------------------------------------:|
| R-50 | dino_r50_4scale |   12   |  49.1  | [config](./dino_r50_4scale_1x_coco.yml) | [model](https://paddledet.bj.bcebos.com/models/dino_r50_4scale_1x_coco.pdparams) |
| R-50 | dino_r50_4scale |   24   |  50.5  | [config](./dino_r50_4scale_2x_coco.yml) | [model](https://paddledet.bj.bcebos.com/models/dino_r50_4scale_2x_coco.pdparams) |

**Notes:**

- DINO is trained on COCO train2017 dataset and evaluated on val2017 results of `mAP(IoU=0.5:0.95)`.
- DINO uses 4GPU to train.

GPU multi-card training
```bash
python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/dino/dino_r50_4scale_1x_coco.yml --fleet --eval
```

## Custom Operator
- Multi-scale deformable attention custom operator see [here](../../ppdet/modeling/transformers/ext_op).

## Citations
```
@misc{zhang2022dino,
      title={DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection},
      author={Hao Zhang and Feng Li and Shilong Liu and Lei Zhang and Hang Su and Jun Zhu and Lionel M. Ni and Heung-Yeung Shum},
      year={2022},
      eprint={2203.03605},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
