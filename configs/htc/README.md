# Hybrid Task Cascade for Instance Segmentation



## Introduction

Cascade is a classic yet powerful architecture that has boosted performance on various tasks. However, how to introduce cascade to instance segmentation remains an open question. A simple combination of Cascade R-CNN and Mask R-CNN only brings limited gain. In this work, authors propose a new framework, Hybrid Task Cascade (HTC), which differs in two important aspects: (1) instead of performing cascaded refinement on these two tasks separately, it interweaves them for a joint multi-stage processing; (2) it adopts a fully convolutional branch to provide spatial context, which can help distinguishing hard foreground from cluttered background. Overall, this framework can learn more discriminative features progressively while integrating complementary features together in each stage. Without bells and whistles, a single HTC obtains 38.4% and 1.5% improvement over a strong Cascade Mask R-CNN baseline on MSCOCO dataset. Moreover, the overall system achieves 48.6 mask AP on the test-challenge split, ranking 1st in the COCO 2018 Challenge Object Detection Task.

## Model Zoo

| Backbone        | Model      | images/GPU | lr schedule |Box AP |Mask AP  |                           download                          | config |
| :-------------- | :------------- | :-----: | :-----: | :------------: | :-----: | :-----------------------------------------------------: | :-----: |
| ResNet50-FPN    | HTC           |    4    |   1x      |     42.1     |  37.3  | [download](https://bj.bcebos.com/v1/paddledet/models/htc_r50_fpn_1x_coco.pdparams) | [config](htc_r50_fpn_1x_coco.yml) |


**Notes:**

- HTC is trained on COCO train2017 dataset and evaluated on val2017 results of `mAP(IoU=0.5:0.95)`.


## Reference
- https://github.com/laihuihui/htc
- https://aistudio.baidu.com/aistudio/projectdetail/2253839

## Citations
```
@article{DBLP:journals/corr/abs-1901-07518,
  author    = {Kai Chen and
               Jiangmiao Pang and
               Jiaqi Wang and
               Yu Xiong and
               Xiaoxiao Li and
               Shuyang Sun and
               Wansen Feng and
               Ziwei Liu and
               Jianping Shi and
               Wanli Ouyang and
               Chen Change Loy and
               Dahua Lin},
  title     = {Hybrid Task Cascade for Instance Segmentation},
  journal   = {CoRR},
  volume    = {abs/1901.07518},
  year      = {2019},
  url       = {http://arxiv.org/abs/1901.07518},
  eprinttype = {arXiv},
  eprint    = {1901.07518},
  timestamp = {Mon, 25 Apr 2022 16:45:48 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1901-07518.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
