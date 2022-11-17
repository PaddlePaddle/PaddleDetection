# ConvNeXt (A ConvNet for the 2020s)

## 模型库
### ConvNeXt on COCO

| 网络网络                  | 输入尺寸 | 图片数/GPU | 学习率策略 | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 | Params(M) | FLOPs(G) |    下载链接       | 配置文件 |
| :------------- | :------- | :-------: | :------: | :------------: | :---------------------: | :----------------: |:---------: | :------: |:---------------: |
| PP-YOLOE-ConvNeXt-tiny | 640 |    16      |   36e    |  44.6  |  63.3 |  33.04  |  13.87 | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_convnext_tiny_36e_coco.pdparams) | [配置文件](./ppyoloe_convnext_tiny_36e_coco.yml) |
| YOLOX-ConvNeXt-s       | 640 |    8       |   36e    |  44.6  |  65.3 |  36.20  |  27.52 | [下载链接](https://paddledet.bj.bcebos.com/models/yolox_convnext_s_36e_coco.pdparams) | [配置文件](./yolox_convnext_s_36e_coco.yml) |


## Citations
```
@Article{liu2022convnet,
  author  = {Zhuang Liu and Hanzi Mao and Chao-Yuan Wu and Christoph Feichtenhofer and Trevor Darrell and Saining Xie},
  title   = {A ConvNet for the 2020s},
  journal = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year    = {2022},
}
```
