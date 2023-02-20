# YOLOF (You Only Look One-level Feature)

## ModelZOO

| 网络网络                | 输入尺寸   | 图片数/GPU | Epochs | 模型推理耗时(ms) | mAP<sup>val<br>0.5:0.95  | Params(M) | FLOPs(G) |    下载链接       | 配置文件 |
| :--------------------- | :------- | :-------: | :----: | :----------: | :---------------------: | :----------------: |:---------: | :------: |:---------------: |
| YOLOF-R_50_C5 (paper)  |  800x1333 |    4     |   12    |      -      |         37.7       |  -  |  - | - | - |
| YOLOF-R_50_C5          |  800x1333 |    4     |   12    |      -      |         38.1       |  44.16  |  241.64 | [下载链接](https://paddledet.bj.bcebos.com/models/yolof_r50_c5_1x_coco.pdparams) | [配置文件](./yolof_r50_c5_1x_coco.yml) |

**注意:**
  - YOLOF模型训练过程中默认使用8 GPUs进行混合精度训练，总batch_size默认为32。


## Citations
```
@inproceedings{chen2021you,
  title={You Only Look One-level Feature},
  author={Chen, Qiang and Wang, Yingming and Yang, Tong and Zhang, Xiangyu and Cheng, Jian and Sun, Jian},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  year={2021}
}
```
