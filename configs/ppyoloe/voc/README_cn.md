# PP-YOLOE

## 模型库

### VOC数据集模型库
|       模型       | Epoch |   GPU个数   | 每GPU图片个数 |  骨干网络  |   输入尺寸   | Box AP<sup>0.5 | Params(M) | FLOPs(G) | V100 FP32(FPS) | V100 TensorRT FP16(FPS) |  模型下载  |  配置文件 |
|:---------------:|:-----:|:-----------:|:-----------:|:---------:|:----------:|:--------------:|:---------:|:---------:|:-------------:|:-----------------------:| :-------: |:--------:|
|   PP-YOLOE+_s   |  30   |     8     |    8     | cspresnet-s |    640     |   86.7  |  7.93  |  17.36   |   208.3   |  333.3   | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_s_30e_voc.pdparams) | [config](./ppyoloe_plus_crn_s_30e_voc.yml) |
|   PP-YOLOE+_l   |  30   |     8     |    8     | cspresnet-l |    640     |   89.0  |  52.20 |  110.07  |   78.1    |  149.2   | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_l_30e_voc.pdparams) | [config](./ppyoloe_plus_crn_l_30e_voc.yml) |
