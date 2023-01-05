# Vision Transformer Detection

## Introduction

- [Context Autoencoder for Self-Supervised Representation Learning](https://arxiv.org/abs/2202.03026)  
- [Benchmarking Detection Transfer Learning with Vision Transformers](https://arxiv.org/pdf/2111.11429.pdf)  

Object detection is a central downstream task used to
test if pre-trained network parameters confer benefits, such
as improved accuracy or training speed. The complexity
of object detection methods can make this benchmarking
non-trivial when new architectures, such as Vision Transformer (ViT) models, arrive.

## Model Zoo

| Model | Backbone | Pretrained | Scheduler | Images/GPU  | Box AP | Mask AP | Config | Download |
|:------:|:--------:|:--------------:|:--------------:|:--------------:|:--------------:|:------:|:------:|:--------:|
| Cascade RCNN | ViT-base | CAE | 1x | 1 | 52.7 | - | [config](./cascade_rcnn_vit_base_hrfpn_cae_1x_coco.yml) | [model](https://bj.bcebos.com/v1/paddledet/models/cascade_rcnn_vit_base_hrfpn_cae_1x_coco.pdparams) |
| Cascade RCNN | ViT-large | CAE | 1x | 1 | 55.7 | - | [config](./cascade_rcnn_vit_large_hrfpn_cae_1x_coco.yml) | [model](https://bj.bcebos.com/v1/paddledet/models/cascade_rcnn_vit_large_hrfpn_cae_1x_coco.pdparams) |
| PP-YOLOE | ViT-base | CAE | 36e | 2 | 52.2 | - | [config](./ppyoloe_vit_base_csppan_cae_36e_coco.yml) | [model](https://bj.bcebos.com/v1/paddledet/models/ppyoloe_vit_base_csppan_cae_36e_coco.pdparams) |
| Mask RCNN | ViT-base | CAE | 1x | 1 | 50.6 | 44.9 | [config](./mask_rcnn_vit_base_hrfpn_cae_1x_coco.yml) | [model](https://bj.bcebos.com/v1/paddledet/models/mask_rcnn_vit_base_hrfpn_cae_1x_coco.pdparams) |
| Mask RCNN | ViT-large | CAE | 1x | 1 | 54.2 | 47.4 | [config](./mask_rcnn_vit_large_hrfpn_cae_1x_coco.yml) | [model](https://bj.bcebos.com/v1/paddledet/models/mask_rcnn_vit_large_hrfpn_cae_1x_coco.pdparams) |


**Notes:**
- Model is trained on COCO train2017 dataset and evaluated on val2017 results of `mAP(IoU=0.5:0.95)
- Base model is trained on 8x32G V100 GPU, large model on 8x80G A100
- The `Cascade RCNN` experiments are based on PaddlePaddle 2.2.2

## Citations
```
@article{chen2022context,
  title={Context autoencoder for self-supervised representation learning},
  author={Chen, Xiaokang and Ding, Mingyu and Wang, Xiaodi and Xin, Ying and Mo, Shentong and Wang, Yunhao and Han, Shumin and Luo, Ping and Zeng, Gang and Wang, Jingdong},
  journal={arXiv preprint arXiv:2202.03026},
  year={2022}
}

@article{DBLP:journals/corr/abs-2111-11429,
  author    = {Yanghao Li and
               Saining Xie and
               Xinlei Chen and
               Piotr Doll{\'{a}}r and
               Kaiming He and
               Ross B. Girshick},
  title     = {Benchmarking Detection Transfer Learning with Vision Transformers},
  journal   = {CoRR},
  volume    = {abs/2111.11429},
  year      = {2021},
  url       = {https://arxiv.org/abs/2111.11429},
  eprinttype = {arXiv},
  eprint    = {2111.11429},
  timestamp = {Fri, 26 Nov 2021 13:48:43 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2111-11429.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

@article{Cai_2019,
   title={Cascade R-CNN: High Quality Object Detection and Instance Segmentation},
   ISSN={1939-3539},
   url={http://dx.doi.org/10.1109/tpami.2019.2956516},
   DOI={10.1109/tpami.2019.2956516},
   journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
   publisher={Institute of Electrical and Electronics Engineers (IEEE)},
   author={Cai, Zhaowei and Vasconcelos, Nuno},
   year={2019},
   pages={1–1}
}
```
