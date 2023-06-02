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
- Objects365数据集下载请参考[objects365官网](http://www.objects365.org/overview.html)。具体种类列表可下载由PaddleDetection团队整理的[objects365_detection_label_list.txt](https://bj.bcebos.com/v1/paddledet/data/objects365/objects365_detection_label_list.txt)并存放在`dataset/objects365/`，每一行即表示第几个种类。inference或导出模型时需要读取到种类数，如果没有标注json文件时，可以进行如下更改`configs/datasets/objects365_detection.yml`：
```
TestDataset:
  !ImageFolder
    # anno_path: annotations/zhiyuan_objv2_val.json
    anno_path: objects365_detection_label_list.txt
    dataset_dir: dataset/objects365/
```
