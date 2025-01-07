# Co-DETR

## Introduction


[DETRs with Collaborative Hybrid Assignments Trainingr.](https://arxiv.org/abs/2211.12860) We reproduced the model of the paper.


## Model Zoo

| Model | Backbone | Epochs | Box AP | Config | Download |
|:------:|:--------:|:--------------:|:------:|:------:|:--------:|
| Co-Deformable-DETR | R50 | 12 | 49.7 | [config](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/configs/co_detr/co_detr_r50_1x_coco.yml) | [model](https://paddledet.bj.bcebos.com/models/co_detr_r50_1x_coco.pdparams) |
| Co-Deformable-DETR | Swin-T | 12 | 51.7 | [config](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/configs/co_detr/co_detr_swin_tiny_1x_coco.yml) | [model](https://paddledet.bj.bcebos.com/models/co_detr_swin_tiny_1x_coco.pdparams) |
| Co-DINO | R50 | 12 | 52.0 | [config](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/configs/co_detr/co_dino_r50_1x_coco.yml) | [model](https://paddledet.bj.bcebos.com/models/co_dino_r50_1x_coco.pdparams) |
| Co-DINO | Swin-L | 12 | 58.7 | [config](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/configs/co_detr/co_dino_swin_large_1x_coco.yml) | [model](https://paddledet.bj.bcebos.com/models/co_dino_swin_large_1x_coco.pdparams) |


GPU multi-card training
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/co_detr/co_detr_r50_1x_coco.yml
```

Evaluate
```bash
export CUDA_VISIBLE_DEVICES=0 #windows和Mac下不需要执行该命令
python tools/eval.py -c configs/co_detr/co_detr_r50_1x_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/co_detr_r50_1x_coco.pdparams
```

## Citations
```
@inproceedings{zong2023detrs,
  title={Detrs with collaborative hybrid assignments training},
  author={Zong, Zhuofan and Song, Guanglu and Liu, Yu},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={6748--6758},
  year={2023}
}
```
