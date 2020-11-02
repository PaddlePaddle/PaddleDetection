# SOLOv2 for instance segmentation

## Introduction

SOLOv2 (Segmenting Objects by Locations) is a fast instance segmentation framework with strong performance. We reproduced the model of the paper, and improved and optimized the accuracy and speed of the SOLOv2.

**Highlights:**

- Performance: `Light-R50-VD-DCN-FPN` model reached 38.6 FPS on single Tesla V100, and mask ap on the COCO-val dataset reached 38.8, which increased inference speed by 24%, mAP increased by 2.4 percentage points.
- Training Time: The training time of the model of `solov2_r50_fpn_1x` on Tesla v100 with 8 GPU is only 10 hours.


## Model Zoo

| Backbone                | Multi-scale training  | Lr schd | Inf time (V100) | Mask AP |         Download                  | Configs |
| :---------------------: | :-------------------: | :-----: | :------------: | :-----: | :---------: | :------------------------: |
| Mobilenetv3-FPN                 |  True                |   3x    |     20.0ms          |  30.0   | [model](https://paddlemodels.bj.bcebos.com/object_detection/solov2_mobilenetv3_fpn_448_3x.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/solov2/solov2_mobilenetv3_fpn_448_3x.yml) |
| R50-FPN                 |  False                |   1x    |     45.7ms          |  35.6   | [model](https://paddlemodels.bj.bcebos.com/object_detection/solov2_r50_fpn_1x.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/solov2/solov2_r50_fpn_1x.yml) |
| R50-FPN                 |  True                |   3x    |     45.7ms          |  37.9   | [model](https://paddlemodels.bj.bcebos.com/object_detection/solov2_r50_fpn_3x.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/solov2/solov2_r50_fpn_3x.yml) |
| R101-VD-FPN                 |  True               |   3x    |     82.6ms       |  42.6   | [model](https://paddlemodels.bj.bcebos.com/object_detection/solov2_r101_vd_fpn_3x.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/solov2/solov2_r101_vd_fpn_3x.yml) |

## Enhanced model
| Backbone                | Input size  | Lr schd | Inf time (V100) | Mask AP |         Download                  | Configs |
| :---------------------: | :-------------------: | :-----: | :------------: | :-----: | :---------: | :------------------------: |
| Light-R50-VD-DCN-FPN          |  512     |   3x    |     25.9ms          |  38.8   | [model](https://paddlemodels.bj.bcebos.com/object_detection/solov2_light_r50_vd_fpn_dcn_512_3x.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/solov2/solov2_light_r50_vd_fpn_dcn_512_3x.yml) |

## Citations
```
@article{wang2020solov2,
  title={SOLOv2: Dynamic, Faster and Stronger},
  author={Wang, Xinlong and Zhang, Rufeng and  Kong, Tao and Li, Lei and Shen, Chunhua},
  journal={arXiv preprint arXiv:2003.10152},
  year={2020}
}
```
