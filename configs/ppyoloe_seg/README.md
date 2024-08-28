# PP-YOLOE Instance segmentation

## 模型库

### 实例分割模型

| 网络网络        | 输入尺寸   | 图片数/GPU | 学习率策略 | TRT-FP16-Latency(ms) | box AP | mask AP | Params(M) | FLOPs(G) |    下载链接       | 配置文件 |
| :------------- | :------- | :-------: | :------: | :------------: | :---------------------: | :----------------: |:---------: | :------: |:---------------: |:-----: |
| PP-YOLOE_seg_s   |  640     |    8      |   80e   |    -   | 42.3 | 32.5 |  8.99   | - | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_seg_s_80e_coco.pdparams) | [配置文件](./ppyoloe_seg_s_80e_coco.yml) |
| PP-YOLOE_seg_m   |  640     |    8      |   80e   |    -   |  -  | - |  26.03   | - | [下载链接]() | [配置文件](./ppyoloe_seg_m_80e_coco.yml) |
| PP-YOLOE_seg_l   |  640     |    8      |   80e   |    -   |  -  | - |  57.32   | - | [下载链接]() | [配置文件](./ppyoloe_seg_l_80e_coco.yml) |
| PP-YOLOE_seg_x   |  640     |    8      |   80e   |    -   |  -  | - |  107.27   | - | [下载链接]() | [配置文件](./ppyoloe_seg_x_80e_coco.yml) |
