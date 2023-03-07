# Group DETR: Fast DETR training with group-wise one-to-many assignment
# Group DETR v2: Strong object detector with encoder-decoder pretraining

## Introduction

[Group DETR](https://arxiv.org/pdf/2207.13085.pdf) is an object detection model based on DETR. We reproduced the model of the paper.

[Group DETR v2](https://arxiv.org/pdf/2211.03594.pdf) is a strong object detection model based on DINO and Group DETR. We reproduced the model of the paper.

## Model Zoo

| Backbone |      Model      | Epochs | Resolution |Box AP |                 Config                  |                                     Download                                     |
|:------:|:---------------:|:------:|:------:|:---------------------------------------:|:--------------------------------------------------------------------------------:|:------:|
| R-50 | dino_r50_4scale |   12   | (800, 1333) |  49.6  | [config](./group_dino_r50_4scale_1x_coco.yml) | [model](https://bj.bcebos.com/v1/paddledet/models/group_dino_r50_4scale_1x_coco.pdparams) |
| Vit-huge | dino_vit_huge_4scale |   12   | (1184, 2000) | 63.3  | [config](./group_dino_vit_huge_4scale_1x_coco.yml) | [model](https://bj.bcebos.com/v1/paddledet/models/group_dino_vit_huge_4scale_1x_coco.pdparams) |

**Notes:**

- Group DETR is trained on COCO train2017 dataset and evaluated on val2017 results of `mAP(IoU=0.5:0.95)`.
- Group DETRv2 requires a ViT-Huge encoder pre-trained and fine-tuned on ImageNet-1K in a self-supervised manner, a detector pre-trained on Object365, and finally it is fine-tuned on trainCOCO. Group DETRv2 is also evaluated on val2017 results of `mAP(IoU=0.5:0.95)`.
- Group DETR and Group DETRv2 are both use 4GPU to train.

GPU multi-card training
```bash
python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/group_detr/group_dino_r50_4scale_1x_coco.yml --fleet --eval
```

```bash
python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/group_detr/group_dino_vit_huge_4scale_1x_coco.yml --fleet --eval
```

## Citations
```
@article{chen2022group,
  title={Group DETR: Fast DETR training with group-wise one-to-many assignment},
  author={Chen, Qiang and Chen, Xiaokang and Wang, Jian and Feng, Haocheng and Han, Junyu and Ding, Errui and Zeng, Gang and Wang, Jingdong},
  journal={arXiv preprint arXiv:2207.13085},
  volume={1},
  number={2},
  year={2022}
}

@article{chen2022group,
  title={Group DETR v2: Strong object detector with encoder-decoder pretraining},
  author={Chen, Qiang and Wang, Jian and Han, Chuchu and Zhang, Shan and Li, Zexian and Chen, Xiaokang and Chen, Jiahui and Wang, Xiaodi and Han, Shuming and Zhang, Gang and others},
  journal={arXiv preprint arXiv:2211.03594},
  year={2022}
}
```
