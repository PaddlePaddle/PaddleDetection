简体中文 | [English](README.md)

# VisDrone-DET 检测模型

PaddleDetection团队提供了针对VisDrone-DET小目标数航拍场景的基于PP-YOLOE的检测模型，用户可以下载模型进行使用。整理后的COCO格式VisDrone-DET数据集[下载链接](https://bj.bcebos.com/v1/paddledet/data/smalldet/visdrone.zip)，检测其中的10类，包括 `pedestrian(1), people(2), bicycle(3), car(4), van(5), truck(6), tricycle(7), awning-tricycle(8), bus(9), motor(10)`，原始数据集[下载链接](https://github.com/VisDrone/VisDrone-Dataset)。

|    模型   | COCOAPI mAP<sup>val<br>0.5:0.95 | COCOAPI mAP<sup>val<br>0.5 | COCOAPI mAP<sup>test_dev<br>0.5:0.95 | COCOAPI mAP<sup>test_dev<br>0.5 | MatlabAPI mAP<sup>test_dev<br>0.5:0.95 | MatlabAPI mAP<sup>test_dev<br>0.5 | 下载  | 配置文件 |
|:---------|:------:|:------:| :----: | :------:| :------: | :------:| :----: | :------:|
|PP-YOLOE-s|  23.5  |  39.9  |  19.4  |  33.6   |  23.68   |  40.66  | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_s_80e_visdrone.pdparams) | [配置文件](./ppyoloe_crn_s_80e_visdrone.yml) |
|PP-YOLOE-l|  29.2  |  47.3  |  23.5  |  39.1   |  28.00   |  46.20  | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_80e_visdrone.pdparams) | [配置文件](./ppyoloe_crn_l_80e_visdrone.yml) |
|PP-YOLOE-P2-l|  30.0  |  49.2  |  24.1  |  40.9   |  28.47   |  48.16  | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_p2_crn_l_80e_visdrone.pdparams) | [配置文件](./ppyoloe_p2_crn_l_80e_visdrone.yml) |
|PP-YOLOE-l largesize|  40.3  |  63.5  |  31.3  |  51.8   |  36.13   |  59.96  | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_80e_visdrone_largesize.pdparams) | [配置文件](./ppyoloe_crn_l_80e_visdrone_largesize.yml) |

**注意:**
- PP-YOLOE模型训练过程中使用8 GPUs进行混合精度训练，如果**GPU卡数**或者**batch size**发生了改变，你需要按照公式 **lr<sub>new</sub> = lr<sub>default</sub> * (batch_size<sub>new</sub> * GPU_number<sub>new</sub>) / (batch_size<sub>default</sub> * GPU_number<sub>default</sub>)** 调整学习率。
- 具体使用教程请参考[ppyoloe](../ppyoloe#getting-start)。
- PP-YOLOE-P2是指增加P2层(1/4下采样层)的特征，共输出4个PPYOLOEHead。
- largesize是指使用以1600尺度为基础的多尺度训练和1920尺度预测，相应的训练batch_size也减小，以速度来换取高精度。
- MatlabAPI测试是使用官网评测工具[VisDrone2018-DET-toolkit](https://github.com/VisDrone/VisDrone2018-DET-toolkit)。


## 引用
```
@ARTICLE{9573394,
  author={Zhu, Pengfei and Wen, Longyin and Du, Dawei and Bian, Xiao and Fan, Heng and Hu, Qinghua and Ling, Haibin},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  title={Detection and Tracking Meet Drones Challenge},
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TPAMI.2021.3119563}
}
```
