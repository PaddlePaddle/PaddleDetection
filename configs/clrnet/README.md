English | [简体中文](README_cn.md)

# CLRNet (CLRNet: Cross Layer Refinement Network for Lane Detection)

## Table of Contents
- [Introduction](#Introduction)
- [Model Zoo](#Model_Zoo)
- [Citations](#Citations)

## Introduction

[CLRNet](https://arxiv.org/abs/2203.10350) is a lane detection model. The CLRNet model is designed with line prior for lane detection, line iou loss as well as nms method, fused to extract contextual high-level features of lane line with low-level features, and refined by FPN multi-scale. Finally, the model achieved SOTA performance in lane detection datasets.

## Model Zoo

### CLRNet Results on CULane dataset

| backbone       | mF1 | F1@50   |    F1@75    | download | config |
| :--------------| :------- |  :----: | :------: | :----: |:-----: |
| ResNet-18         | 54.98 |  79.46  |    62.10   | [model](https://paddledet.bj.bcebos.com/models/clrnet_resnet18_culane.pdparams) | [config](./clrnet_resnet18_culane.yml) |

### Download
Download [CULane](https://xingangpan.github.io/projects/CULane.html). Then extract them to `dataset/culane`.

For CULane, you should have structure like this:
```shell
culane/driver_xx_xxframe    # data folders x6
culane/laneseg_label_w16    # lane segmentation labels
culane/list                 # data lists
```
If you use Baidu Cloud, make sure that images in `driver_23_30frame_part1.tar.gz` and `driver_23_30frame_part2.tar.gz` are located in one folder `driver_23_30frame` instead of two seperate folders after you decompress them.

Now we have uploaded a small subset of CULane dataset to PaddleDetection for code checking. You can simply run the training script below to download it automatically. If you want to implement the results, you need to download the full dataset at th link for training.

### Training
- single GPU
```shell
python tools/train.py -c configs/clrnet/clr_resnet18_culane.yml
```
- multi GPU
```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/clrnet/clr_resnet18_culane.yml
```

### Evaluation
```shell
python tools/eval.py -c configs/clrnet/clr_resnet18_culane.yml -o weights=output/clr_resnet18_culane/model_final.pdparams
```

### Inference
```shell
python tools/infer_culane.py -c configs/clrnet/clr_resnet18_culane.yml -o weights=output/clr_resnet18_culane/model_final.pdparams --infer_img=demo/lane00000.jpg
```

Notice: The inference phase does not support static model graph deploy at present.

## Citations
```
@InProceedings{Zheng_2022_CVPR,
    author    = {Zheng, Tu and Huang, Yifei and Liu, Yang and Tang, Wenjian and Yang, Zheng and Cai, Deng and He, Xiaofei},
    title     = {CLRNet: Cross Layer Refinement Network for Lane Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {898-907}
}
```
