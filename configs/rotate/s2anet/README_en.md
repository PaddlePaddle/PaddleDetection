English | [简体中文](README.md)

# S2ANet

## Content
- [Introduction](#Introduction)
- [Model Zoo](#Model-Zoo)
- [Getting Start](#Getting-Start)
- [Deployment](#Deployment)
- [Citations](#Citations)

## Introduction

[S2ANet](https://arxiv.org/pdf/2008.09397.pdf) is used to detect rotated objects.

## Model Zoo
| Model | Conv Type | mAP | Lr Scheduler | Angle | Aug | GPU Number | images/GPU | download | config |
|:---:|:------:|:----:|:---------:|:-----:|:--------:|:-----:|:------------:|:-------:|:------:|
| S2ANet | Conv | 71.45 | 2x | le135 | - | 4 | 2 | [model](https://paddledet.bj.bcebos.com/models/s2anet_conv_2x_dota.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/rotate/s2anet/s2anet_conv_2x_dota.yml) |
| S2ANet | AlignConv | 73.84 | 2x | le135 | - | 4 | 2 | [model](https://paddledet.bj.bcebos.com/models/s2anet_alignconv_2x_dota.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/rotate/s2anet/s2anet_alignconv_2x_dota.yml) |

**Notes:**
- if **GPU number** or **mini-batch size** is changed, **learning rate** should be adjusted according to the formula **lr<sub>new</sub> = lr<sub>default</sub> * (batch_size<sub>new</sub> * GPU_number<sub>new</sub>) / (batch_size<sub>default</sub> * GPU_number<sub>default</sub>)**.
- Models in model zoo is trained and tested with single scale by default. If `MS` is indicated in the data augmentation column, it means that multi-scale training and multi-scale testing are used. If `RR` is indicated in the data augmentation column, it means that RandomRotate data augmentation is used for training.
- `multiclass_nms` is used here, which is slightly different from the original author's use of NMS.

## Getting Start

Refer to [Data-Preparation](../README_en.md#Data-Preparation) to prepare data.

### 1. Train

Single GPU Training
```bash
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/rotate/s2anet/s2anet_1x_spine.yml
```

Multiple GPUs Training
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/rotate/s2anet/s2anet_1x_spine.yml
```

You can use `--eval`to enable train-by-test.

### 2. Evaluation
```bash
python tools/eval.py -c configs/rotate/s2anet/s2anet_1x_spine.yml -o weights=output/s2anet_1x_spine/model_final.pdparams

# Use a trained model to evaluate
python tools/eval.py -c configs/rotate/s2anet/s2anet_1x_spine.yml -o weights=https://paddledet.bj.bcebos.com/models/s2anet_1x_spine.pdparams
```

### 3. Prediction
Executing the following command will save the image prediction results to the `output` folder.
```bash
python tools/infer.py -c configs/rotate/s2anet/s2anet_1x_spine.yml -o weights=output/s2anet_1x_spine/model_final.pdparams --infer_img=demo/39006.jpg --draw_threshold=0.3
```
Prediction using models that provide training:
```bash
python tools/infer.py -c configs/rotate/s2anet/s2anet_1x_spine.yml -o weights=https://paddledet.bj.bcebos.com/models/s2anet_1x_spine.pdparams --infer_img=demo/39006.jpg --draw_threshold=0.3
```

### 4. DOTA Data evaluation
Execute the following command, will save each image prediction result in `output` folder txt text with the same folder name.
```
python tools/infer.py -c configs/rotate/s2anet/s2anet_alignconv_2x_dota.yml -o weights=https://paddledet.bj.bcebos.com/models/s2anet_alignconv_2x_dota.pdparams --infer_dir=/path/to/test/images --output_dir=output --visualize=False --save_results=True
```
Refering to [DOTA Task](https://captain-whu.github.io/DOTA/tasks.html), You need to submit a zip file containing results for all test images for evaluation. The detection results of each category are stored in a txt file, each line of which is in the following format
`image_id score x1 y1 x2 y2 x3 y3 x4 y4`. To evaluate, you should submit the generated zip file to the Task1 of [DOTA Evaluation](https://captain-whu.github.io/DOTA/evaluation.html). You can execute the following command to generate the file
```
python configs/rotate/tools/generate_result.py --pred_txt_dir=output/ --output_dir=submit/ --data_type=dota10

zip -r submit.zip submit
```

## Deployment

The inputs of the `multiclass_nms` operator in Paddle support quadrilateral inputs, so deployment can be done without relying on the rotating frame IOU operator.

Please refer to the deployment tutorial[Predict deployment](../../../deploy/README_en.md)


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
