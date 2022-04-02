# Cascade R-CNN: High Quality Object Detection and Instance Segmentation

## Model Zoo

| 骨架网络             | 网络类型       | 每张GPU图片个数 | 学习率策略 |推理时间(fps) | Box AP | Mask AP |                           下载                          | 配置文件 |
| :------------------- | :------------- | :-----: | :-----: | :------------: | :-----: | :-----: | :-----------------------------------------------------: | :-----: |
| ResNet50-FPN         | Cascade Faster         |    1    |   1x    |     ----     |  41.1  |    -    | [下载链接](https://paddledet.bj.bcebos.com/models/cascade_rcnn_r50_fpn_1x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.yml) |
| ResNet50-FPN         | Cascade Mask         |    1    |   1x    |     ----     |  41.8  |    36.3    | [下载链接](https://paddledet.bj.bcebos.com/models/cascade_mask_rcnn_r50_fpn_1x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.yml) |
| ResNet50-vd-SSLDv2-FPN | Cascade Faster         |    1    |   1x    |     ----     |  44.4  |    -    | [下载链接](https://paddledet.bj.bcebos.com/models/cascade_rcnn_r50_vd_fpn_ssld_1x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/cascade_rcnn/cascade_rcnn_r50_vd_fpn_ssld_1x_coco.yml) |
| ResNet50-vd-SSLDv2-FPN | Cascade Faster         |    1    |   2x    |     ----     |  45.0  |    -    | [下载链接](https://paddledet.bj.bcebos.com/models/cascade_rcnn_r50_vd_fpn_ssld_2x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/cascade_rcnn/cascade_rcnn_r50_vd_fpn_ssld_2x_coco.yml) |
| ResNet50-vd-SSLDv2-FPN | Cascade Mask         |    1    |   1x    |     ----     |  44.9  |    39.1    | [下载链接](https://paddledet.bj.bcebos.com/models/cascade_mask_rcnn_r50_vd_fpn_ssld_1x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/cascade_rcnn/cascade_mask_rcnn_r50_vd_fpn_ssld_1x_coco.yml) |
| ResNet50-vd-SSLDv2-FPN | Cascade Mask         |    1    |   2x    |     ----     |  45.7  |    39.7    | [下载链接](https://paddledet.bj.bcebos.com/models/cascade_mask_rcnn_r50_vd_fpn_ssld_2x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/cascade_rcnn/cascade_mask_rcnn_r50_vd_fpn_ssld_2x_coco.yml) |


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
