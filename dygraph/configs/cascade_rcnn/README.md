# Cascade R-CNN: High Quality Object Detection and Instance Segmentation

## Model Zoo

| 骨架网络             | 网络类型       | 每张GPU图片个数 | 学习率策略 |推理时间(fps) | Box AP | Mask AP |                           下载                          | 配置文件 |
| :------------------- | :------------- | :-----: | :-----: | :------------: | :-----: | :-----: | :-----------------------------------------------------: | :-----: |
| ResNet50-FPN         | Cascade Faster         |    1    |   1x    |     ----     |  41.1  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/dygraph/cascade_rcnn_r50_fpn_1x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/dygraph/configs/cascade_faster_rcnn_r50_fpn_1x_coco.yml) |
| ResNet50-FPN         | Cascade Mask         |    1    |   1x    |     ----     |  41.6  |    35.3    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/dygraph/cascade_mask_rcnn_r50_fpn_1x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/dygraph/configs/cascade_mask_rcnn_r50_fpn_1x_coco.yml) |

## Citations
```
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
