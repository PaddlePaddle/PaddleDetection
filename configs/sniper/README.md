English | [简体中文](README_cn.md)

# SNIPER: Efficient Multi-Scale Training

## Model Zoo

| sniper   | GPU number    | images/GPU |    Model  |    Network     | schedulers | Box AP |          download                  | config |
| :---------------- | :-------------------: | :------------------: | :-----: | :-----: | :------------: | :-----: | :-----------------------------------------------------: | :-----: |
| w/o sniper   |    4    |    1    | ResNet-r50-FPN      | Faster Rcnn         |   1x    |  23.3  | [faster_rcnn_r50_fpn_1x_visdrone](https://bj.bcebos.com/v1/paddledet/models/faster_rcnn_r50_fpn_1x_visdrone.pdparams ) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/sniper/faster_rcnn_r50_fpn_1x_sniper_coco.yml) |
| w sniper |    4    |    1    | ResNet-r50-FPN      | Faster Rcnn         |   1x    |  29.7  | [faster_rcnn_r50_fpn_1x_sniper_visdrone](https://bj.bcebos.com/v1/paddledet/models/faster_rcnn_r50_fpn_1x_sniper_visdrone.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/sniper/faster_rcnn_r50_fpn_2x_sniper_coco.yml) |

## Getting Start
### 1. Training
a.optional: Run `tools/sniper_params_stats.py` to get image_target_sizes\valid_box_ratio_ranges\chip_target_size\chip_target_stride，and modify this params in configs/datasets/sniper_coco_detection.yml
```bash
python tools/sniper_params_stats.py FasterRCNN annotations/instances_train2017.json
```
b.optional: trian detector to get negative proposals.
```bash
python -m paddle.distributed.launch --log_dir=./sniper/ --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/sniper/faster_rcnn_r50_fpn_2x_sniper_coco.yml --save_proposals --proposals_path=./proposals.json &>sniper.log 2>&1 &
```
c.train models
```bash
python -m paddle.distributed.launch --log_dir=./sniper/ --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/sniper/faster_rcnn_r50_fpn_2x_sniper_coco.yml --eval &>sniper.log 2>&1 &
```

### 2. Evaluation
Evaluating SNIPER on custom dataset in single GPU with following commands:
```bash
# use saved checkpoint in training
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/sniper/faster_rcnn_r50_fpn_2x_sniper_coco.yml -o weights=output/faster_rcnn_r50_fpn_2x_sniper_coco/model_final
```

###3.Inference
Inference images in single GPU with following commands, use `--infer_img` to inference a single image and `--infer_dir` to inference all images in the directory.

```bash
# inference single image
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/sniper/faster_rcnn_r50_fpn_2x_sniper_coco.yml -o weights=output/faster_rcnn_r50_fpn_2x_sniper_coco/model_final --infer_img=demo/P0861__1.0__1154___824.png

# inference all images in the directory
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/sniper/faster_rcnn_r50_fpn_2x_sniper_coco.yml -o weights=output/faster_rcnn_r50_fpn_2x_sniper_coco/model_final --infer_dir=demo
```

## Citations
@misc{1805.09300,
Author = {Bharat Singh and Mahyar Najibi and Larry S. Davis},
Title = {SNIPER: Efficient Multi-Scale Training},
Year = {2018},
Eprint = {arXiv:1805.09300},
}
