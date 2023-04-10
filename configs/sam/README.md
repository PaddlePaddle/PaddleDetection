# SAM (Segment Anything)

## Model Zoo

| 网络网络                | 输入尺寸   | 图片数/GPU | Epochs | 模型推理耗时(ms) | mAP<sup>val<br>0.5:0.95  | Params(M) | FLOPs(G) |    下载链接       | 配置文件 |
| :--------------------- | :------- | :-------: | :----: | :----------: | :---------------------: | :----------------: |:---------: | :------: |:---------------: |
| SAM-B         |  1024x1024 |    -     |   -    |      -      |         -       |  -  |  - | [下载链接](https://paddledet.bj.bcebos.com/models/sam_vit_b_coco.pdparams) | [配置文件](./sam_vit_b_coco.yml) |
| SAM-L         |  1024x1024 |    -     |   -    |      -      |         -       |  -  |  - | [下载链接](https://paddledet.bj.bcebos.com/models/sam_vit_l_coco.pdparams) | [配置文件](./sam_vit_l_coco.yml) |
| SAM-H         |  1024x1024 |    -     |   -    |      -      |         -       |  -  |  - | [下载链接](https://paddledet.bj.bcebos.com/models/sam_vit_h_coco.pdparams) | [配置文件](./sam_vit_h_coco.yml) |

**注意:**
  - SAM 仅支持推理


## Citations
```
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```
