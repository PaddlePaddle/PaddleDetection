# S2ANet Model

## Content
- [S2ANet Model](#s2anet-model)
  - [Content](#content)
  - [Introduction](#introduction)
  - [Start Training](#start-training)
    - [1. Train](#1-train)
    - [2. Evaluation](#2-evaluation)
    - [3. Prediction](#3-prediction)
    - [4. DOTA Data evaluation](#4-dota-data-evaluation)
  - [Model Library](#model-library)
    - [S2ANet Model](#s2anet-model-1)
  - [Predict Deployment](#predict-deployment)
  - [Citations](#citations)

## Introduction

[S2ANet](https://arxiv.org/pdf/2008.09397.pdf) is used to detect rotated objects and acheives 74.0 mAP on DOTA 1.0 dataset.

## Start Training

### 2. Train

Single GPU Training
```bash
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/rotate/s2anet/s2anet_1x_spine.yml
```

Multiple GPUs Training
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/rotate/s2anet/s2anet_1x_spine.yml
```

You can use `--eval`to enable train-by-test.

### 3. Evaluation
```bash
python tools/eval.py -c configs/rotate/s2anet/s2anet_1x_spine.yml -o weights=output/s2anet_1x_spine/model_final.pdparams

# Use a trained model to evaluate
python tools/eval.py -c configs/rotate/s2anet/s2anet_1x_spine.yml -o weights=https://paddledet.bj.bcebos.com/models/s2anet_1x_spine.pdparams
```

### 4. Prediction
Executing the following command will save the image prediction results to the `output` folder.
```bash
python tools/infer.py -c configs/rotate/s2anet/s2anet_1x_spine.yml -o weights=output/s2anet_1x_spine/model_final.pdparams --infer_img=demo/39006.jpg --draw_threshold=0.3
```
Prediction using models that provide training:
```bash
python tools/infer.py -c configs/rotate/s2anet/s2anet_1x_spine.yml -o weights=https://paddledet.bj.bcebos.com/models/s2anet_1x_spine.pdparams --infer_img=demo/39006.jpg --draw_threshold=0.3
```

### 5. DOTA Data evaluation
Execute the following command, will save each image prediction result in `output` folder txt text with the same folder name.
```
python tools/infer.py -c configs/rotate/s2anet/s2anet_alignconv_2x_dota.yml -o weights=./weights/s2anet_alignconv_2x_dota.pdparams --infer_dir=/path/to/test/images --output_dir=output --visualize=False --save_results=True
```
Refering to [DOTA Task](https://captain-whu.github.io/DOTA/tasks.html), You need to submit a zip file containing results for all test images for evaluation. The detection results of each category are stored in a txt file, each line of which is in the following format
`image_id score x1 y1 x2 y2 x3 y3 x4 y4`. To evaluate, you should submit the generated zip file to the Task1 of [DOTA Evaluation](https://captain-whu.github.io/DOTA/evaluation.html). You can execute the following command to generate the file
```
python configs/rotate/tools/generate_result.py --pred_txt_dir=output/ --output_dir=submit/ --data_type=dota10
zip -r submit.zip submit
```

## Model Library

### S2ANet Model

|     Model     |  Conv Type  |   mAP    |   Model Download   |   Configuration File   |
|:-----------:|:----------:|:--------:| :----------:| :---------: |
|   S2ANet    |   Conv     |   71.42  |  [model](https://paddledet.bj.bcebos.com/models/s2anet_conv_2x_dota.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/rotate/s2anet/s2anet_conv_2x_dota.yml)                   |
|   S2ANet    |  AlignConv |   74.0   |  [model](https://paddledet.bj.bcebos.com/models/s2anet_alignconv_2x_dota.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/rotate/s2anet/s2anet_alignconv_2x_dota.yml)                   |

**Attention:** `multiclass_nms` is used here, which is slightly different from the original author's use of NMS.


## Predict Deployment

The inputs of the `multiclass_nms` operator in Paddle support quadrilateral inputs, so deployment can be done without relying on the rotating frame IOU operator.

Please refer to the deployment tutorial[Predict deployment](../../deploy/README_en.md)


## Citations
```
@article{han2021align,  
  author={J. {Han} and J. {Ding} and J. {Li} and G. -S. {Xia}},  
  journal={IEEE Transactions on Geoscience and Remote Sensing},  
  title={Align Deep Features for Oriented Object Detection},  
  year={2021},
  pages={1-11},  
  doi={10.1109/TGRS.2021.3062048}}

@inproceedings{xia2018dota,
  title={DOTA: A large-scale dataset for object detection in aerial images},
  author={Xia, Gui-Song and Bai, Xiang and Ding, Jian and Zhu, Zhen and Belongie, Serge and Luo, Jiebo and Datcu, Mihai and Pelillo, Marcello and Zhang, Liangpei},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3974--3983},
  year={2018}
}
```
