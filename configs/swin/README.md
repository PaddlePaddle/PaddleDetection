# Swin Transformer

## COCO Model Zoo

| 骨架网络             | 网络类型       | 每张GPU图片个数 | 学习率策略 |推理时间(fps) | Box AP |                           下载                          | 配置文件 |
| :------------------- | :------------- | :-----: | :-----: | :------------: | :-----: | :-----------------------------------------------------: | :-----: |
| swin_T_224 | Faster R-CNN    |    2    |   36e    |     ----     |  45.3  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_swin_tiny_fpn_3x_coco.pdparams) | [配置文件](./faster_rcnn_swin_tiny_fpn_3x_coco.yml) |
| swin_T_224 | PP-YOLOE+       |    8    |   36e    |     ----     |  44.7  | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_swin_tiny_36e_coco.pdparams) | [配置文件](./ppyoloe_plus_swin_tiny_36e_coco.yml) |


## Citations
```
@article{liu2021Swin,
    title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
    author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
    journal={arXiv preprint arXiv:2103.14030},
    year={2021}
}

@inproceedings{liu2021swinv2,
  title={Swin Transformer V2: Scaling Up Capacity and Resolution},
  author={Ze Liu and Han Hu and Yutong Lin and Zhuliang Yao and Zhenda Xie and Yixuan Wei and Jia Ning and Yue Cao and Zheng Zhang and Li Dong and Furu Wei and Baining Guo},
  booktitle={International Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```
