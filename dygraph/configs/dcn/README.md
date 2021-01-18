### Deformable ConvNets v2

| 骨架网络             | 网络类型           | 卷积    | 每张GPU图片个数 | 学习率策略 |推理时间(fps)| Box AP | Mask AP |                           下载                           | 配置文件 |
| :------------------- | :------------- | :-----: |:--------: | :-----: | :-----------: |:----: | :-----: | :----------------------------------------------------------: | :----: |
| ResNet50-FPN         | Faster         | c3-c5   |    2      |   1x    |    -     |  41.3  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/dygraph/faster_rcnn_dcn_r50_fpn_1x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/dcn/faster_rcnn_dcn_r50_fpn_1x_coco.yml) |
| ResNet50-vd-FPN      | Faster         | c3-c5   |    2      |   2x    |    -     |  42.4  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/dygraph/faster_rcnn_dcn_r50_vd_fpn_2x.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/dcn/faster_rcnn_dcn_r50_vd_fpn_2x.yml) |
| ResNet101-vd-FPN     | Faster         | c3-c5   |    2      |   1x    |    -     |  44.1  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/dygraph/faster_rcnn_dcn_r101_vd_fpn_1x.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/dcn/faster_rcnn_dcn_r101_vd_fpn_1x.yml) |
| ResNeXt101-vd-FPN    | Faster         | c3-c5   |    1      |   1x    |    -     |  45.2  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/dygraph/faster_rcnn_dcn_x101_vd_64x4d_fpn_1x.pdparams) |[配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/dcn/faster_rcnn_dcn_x101_vd_64x4d_fpn_1x.yml) |
| ResNet50-FPN         | Mask           | c3-c5   |    1      |   1x    |    -     |  41.9  |  37.3   | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/dygraph/mask_rcnn_dcn_r50_fpn_1x.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/dcn/mask_rcnn_dcn_r50_fpn_1x.yml) |
| ResNet50-vd-FPN      | Mask           | c3-c5   |    1      |   2x    |    -     |  42.9  |  38.0   | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/dygraph/mask_rcnn_dcn_r50_vd_fpn_2x.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/dcn/mask_rcnn_dcn_r50_vd_fpn_2x.yml) |
| ResNet101-vd-FPN     | Mask           | c3-c5   |    1      |   1x    |    -     |  44.6  |  39.2   | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/dygraph/mask_rcnn_dcn_r101_vd_fpn_1x.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/dcn/mask_rcnn_dcn_r101_vd_fpn_1x.yml) |
| ResNeXt101-vd-FPN    | Mask           | c3-c5   |    1      |   1x    |     -    |  46.2  |  40.4   | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/dygraph/mask_rcnn_dcn_x101_vd_64x4d_fpn_1x.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/dcn/mask_rcnn_dcn_x101_vd_64x4d_fpn_1x.yml) |

**注意事项:**  

- Deformable卷积网络v2(dcn_v2)参考自论文[Deformable ConvNets v2](https://arxiv.org/abs/1811.11168).
- `c3-c5`意思是在resnet模块的3到5阶段增加`dcn`.

## Citations
```
@inproceedings{dai2017deformable,
  title={Deformable Convolutional Networks},
  author={Dai, Jifeng and Qi, Haozhi and Xiong, Yuwen and Li, Yi and Zhang, Guodong and Hu, Han and Wei, Yichen},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  year={2017}
}
@article{zhu2018deformable,
  title={Deformable ConvNets v2: More Deformable, Better Results},
  author={Zhu, Xizhou and Hu, Han and Lin, Stephen and Dai, Jifeng},
  journal={arXiv preprint arXiv:1811.11168},
  year={2018}
}
```
