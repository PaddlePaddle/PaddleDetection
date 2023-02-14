English | [简体中文](README.md)

# FCOSR

## Content
- [Introduction](#Introduction)
- [Model Zoo](#Model-Zoo)
- [Getting Start](#Getting-Start)
- [Deployment](#Deployment)
- [Citations](#Citations)

## Introduction

[FCOSR](https://arxiv.org/abs/2111.10780) is one stage anchor-free model based on [FCOS](https://arxiv.org/abs/1904.01355). FCOSR focuses on the label assignment strategy for oriented bounding boxes and proposes ellipse center sampling method and fuzzy sample assignment strategy. In terms of loss, FCOSR uses [ProbIoU](https://arxiv.org/abs/2106.06072) to avoid boundary discontinuity problem.

## Model Zoo

| Model | Backbone | mAP | Lr Scheduler | Angle | Aug | GPU Number | images/GPU | download | config |
|:---:|:--------:|:----:|:---------:|:-----:|:--------:|:-----:|:------------:|:-------:|:------:|
| FCOSR-M | ResNeXt-50 | 76.62 | 3x | oc | RR | 4 | 4 | [model](https://paddledet.bj.bcebos.com/models/fcosr_x50_3x_dota.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/rotate/fcosr/fcosr_x50_3x_dota.yml) |

**Notes:**

- if **GPU number** or **mini-batch size** is changed, **learning rate** should be adjusted according to the formula **lr<sub>new</sub> = lr<sub>default</sub> * (batch_size<sub>new</sub> * GPU_number<sub>new</sub>) / (batch_size<sub>default</sub> * GPU_number<sub>default</sub>)**.
- Models in model zoo is trained and tested with single scale by default. If `MS` is indicated in the data augmentation column, it means that multi-scale training and multi-scale testing are used. If `RR` is indicated in the data augmentation column, it means that RandomRotate data augmentation is used for training.

## Getting Start

Refer to [Data-Preparation](../README_en.md#Data-Preparation) to prepare data.

### Training

Single GPU Training
``` bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/rotate/fcosr/fcosr_x50_3x_dota.yml
```

Multiple GPUs Training
``` bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/rotate/fcosr/fcosr_x50_3x_dota.yml
```

### Inference

Run the follow command to infer single image, the result of inference will be saved in `output` directory by default.

``` bash
python tools/infer.py -c configs/rotate/fcosr/fcosr_x50_3x_dota.yml -o weights=https://paddledet.bj.bcebos.com/models/fcosr_x50_3x_dota.pdparams --infer_img=demo/P0861__1.0__1154___824.png --draw_threshold=0.5
```

### Evaluation on DOTA Dataset
Refering to [DOTA Task](https://captain-whu.github.io/DOTA/tasks.html), You need to submit a zip file containing results for all test images for evaluation. The detection results of each category are stored in a txt file, each line of which is in the following format
`image_id score x1 y1 x2 y2 x3 y3 x4 y4`. To evaluate, you should submit the generated zip file to the Task1 of [DOTA Evaluation](https://captain-whu.github.io/DOTA/evaluation.html). You can run the following command to get the inference results of test dataset:
``` bash
python tools/infer.py -c configs/rotate/fcosr/fcosr_x50_3x_dota.yml -o weights=https://paddledet.bj.bcebos.com/models/fcosr_x50_3x_dota.pdparams --infer_dir=/path/to/test/images --output_dir=output_fcosr --visualize=False --save_results=True
```
Process the prediction results into the format required for the official website evaluation:
``` bash
python configs/rotate/tools/generate_result.py --pred_txt_dir=output_fcosr/ --output_dir=submit/ --data_type=dota10

zip -r submit.zip submit
```

## Deployment

Please refer to the deployment tutorial[Deployment](../../../deploy/README_en.md)

## Citations

```
@article{li2021fcosr,
  title={Fcosr: A simple anchor-free rotated detector for aerial object detection},
  author={Li, Zhonghua and Hou, Biao and Wu, Zitong and Jiao, Licheng and Ren, Bo and Yang, Chen},
  journal={arXiv preprint arXiv:2111.10780},
  year={2021}
}

@inproceedings{tian2019fcos,
  title={Fcos: Fully convolutional one-stage object detection},
  author={Tian, Zhi and Shen, Chunhua and Chen, Hao and He, Tong},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={9627--9636},
  year={2019}
}

@article{llerena2021gaussian,
  title={Gaussian Bounding Boxes and Probabilistic Intersection-over-Union for Object Detection},
  author={Llerena, Jeffri M and Zeni, Luis Felipe and Kristen, Lucas N and Jung, Claudio},
  journal={arXiv preprint arXiv:2106.06072},
  year={2021}
}
```
