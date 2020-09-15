# SOLOv2 (Segmenting Objects by Locations) for instance segmentation

## Introduction

- SOLOv2 is a fast instance segmentation framework with strong performance: [https://arxiv.org/abs/2003.10152](https://arxiv.org/abs/2003.10152)

```
@misc{wang2020solov2,
    title={SOLOv2: Dynamic, Faster and Stronger},
    author={Xinlong Wang and Rufeng Zhang and Tao Kong and Lei Li and Chunhua Shen},
    year={2020},
    eprint={2003.10152},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Model Zoo

| Backbone                | Multi-scale training  | Lr schd | Inf time (fps) | Mask AP |         Download                  | Configs |
| :---------------------: | :-------------------: | :-----: | :------------: | :-----: | :---------: | :------------------------: |
| R50-FPN                 |  False                |   1x    |     -          |  34.7   | [model](https://paddlemodels.bj.bcebos.com/object_detection/solov2_r50_fpn_1x.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/solov2/solov2_r50_fpn_1x.yml) |
