# OV_DETR

## Introduction


OV_DETR is a open-vocabulary object detection model based on transformer. We reproduced the model of the paper.


## Model Zoo

| Backbone | Model | Images/GPU  | Inf time (fps) | Box AP | Box AP Seen| Box AP Unseen| Config |
|:------:|:--------:|:--------:|:--------------:|:------:|:------:||:------:||:------:|:--------:|
| R-50 | OV_DETR  | -- | -- | -- | -- | -- |[config](ov_detr_r50_1x_coco.yml) |

## Prepare
Download the open-vocabulary [Annotations](https://bj.bcebos.com/v1/paddledet/data/coco/zero-shot.zip),replace the relevant path in the configuration file
```
ov_detr_r50_1x_coco.yml
  ....
  text_embedding: zeroshot_w.npy
  ....
  clip_feat_path: clip_feat_coco.pkl
_base_/ov_detr_coco_detection.yml
  TrainDataset:
    ....
    anno_path: instances_train2017_seen_2_proposal.json

  EvalDataset:
    ....
    anno_path: instances_val2017_all.json
    ....
```

GPU multi-card training
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/ov_detr/ov_detr_r50_1x_coco.yml
```


## Citations
```
@InProceedings{zang2022open,
 author = {Zang, Yuhang and Li, Wei and Zhou, Kaiyang and Huang, Chen and Loy, Chen Change},
 title = {Open-Vocabulary DETR with Conditional Matching},
 journal = {European Conference on Computer Vision},
 year = {2022}
}
```
