# PP-YOLOE

## 模型库

### Objects365数据集模型库
|       模型       | Epoch |  机器个数 | GPU个数   | 每GPU图片个数 |  骨干网络  |   输入尺寸   | Box AP<sup>0.5 | Params(M) | FLOPs(G) | V100 FP32(FPS) | V100 TensorRT FP16(FPS) |  模型下载  | 配置文件 |
|:---------------:|:-----:|:-----------:|:-----------:|:-----------:|:---------:|:----------:|:--------------:|:---------:|:---------:|:-------------:|:-----------------------:| :--------:|:--------:|
|   PP-YOLOE+_s   |  60   |  3 |  8     |    8     | cspresnet-s |    640     |   18.1  |  7.93  |  17.36   |   208.3   |  333.3   | [model](https://bj.bcebos.com/v1/paddledet/models/pretrained/ppyoloe_crn_s_obj365_pretrained.pdparams) | [config](./ppyoloe_plus_crn_s_60e_objects365.yml) |
|   PP-YOLOE+_m   |  60   |   4 |  8     |    8     | cspresnet-m |    640     |   25.0  |  23.43  |  49.91   |   123.4       |  208.3  | [model](https://bj.bcebos.com/v1/paddledet/models/pretrained/ppyoloe_crn_m_obj365_pretrained.pdparams) | [config](./ppyoloe_plus_crn_m_60e_objects365.yml) |
|   PP-YOLOE+_l   |  60   |   3 |  8     |    8     | cspresnet-l |    640     |   30.8  |  52.20 |  110.07  |   78.1    |  149.2   | [model](https://bj.bcebos.com/v1/paddledet/models/pretrained/ppyoloe_crn_l_obj365_pretrained.pdparams) | [config](./ppyoloe_plus_crn_l_60e_objects365.yml) |
|   PP-YOLOE+_x   |  60   |  4 |   8     |    8     | cspresnet-x |    640     |   32.7  |  98.42 |  206.59      |   45.0        |  95.2  | [model](https://bj.bcebos.com/v1/paddledet/models/pretrained/ppyoloe_crn_x_obj365_pretrained.pdparams) | [config](./ppyoloe_plus_crn_x_60e_objects365.yml) |


**注意:**
- 多机训练细节见[文档](../../../docs/tutorials/DistributedTraining_cn.md)
