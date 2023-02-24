# Sparse R-CNN: End-to-End Object Detection with Learnable Proposals


## Introduction
Sparse RCNN is a purely sparse method for object detection in images.


## Model Zoo

| Backbone        | Proposals | lr schedule | Box AP | download   | config |
| :-------------- | :-----: | :------------: | :-----: | :-----: | :-----: |
| ResNet50-FPN | 100 | 3x |  43.0  | [download](https://paddledet.bj.bcebos.com/models/sparse_rcnn_r50_fpn_3x_pro100_coco.pdparams) | [config](./sparse_rcnn_r50_fpn_3x_pro100_coco.yml) |
| ResNet50-FPN | 300 | 3x |  44.6  | [download](https://paddledet.bj.bcebos.com/models/sparse_rcnn_r50_fpn_3x_pro300_coco.pdparams) | [config](./sparse_rcnn_r50_fpn_3x_pro300_coco.yml) |

## Citations
```
@misc{sun2021sparse,
      title={Sparse R-CNN: End-to-End Object Detection with Learnable Proposals},
      author={Peize Sun and Rufeng Zhang and Yi Jiang and Tao Kong and Chenfeng Xu and Wei Zhan and Masayoshi Tomizuka and Lei Li and Zehuan Yuan and Changhu Wang and Ping Luo},
      year={2021},
      eprint={2011.12450},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
