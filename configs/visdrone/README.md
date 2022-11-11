# VisDrone-DET 检测模型

PaddleDetection团队提供了针对VisDrone-DET小目标数航拍场景的基于PP-YOLOE的检测模型，用户可以下载模型进行使用。整理后的COCO格式VisDrone-DET数据集[下载链接](https://bj.bcebos.com/v1/paddledet/data/smalldet/visdrone.zip)，检测其中的10类，包括 `pedestrian(1), people(2), bicycle(3), car(4), van(5), truck(6), tricycle(7), awning-tricycle(8), bus(9), motor(10)`，原始数据集[下载链接](https://github.com/VisDrone/VisDrone-Dataset)。

**注意:**
- VisDrone-DET数据集包括train集6471张，val集548张，test_dev集1610张，test-challenge集1580张(未开放检测框标注)，前三者均有开放检测框标注。
- 模型均只使用train集训练，在val集和test_dev集上验证精度，test_dev集图片数较多，精度参考性较高。


## 原图训练：

|    模型   | COCOAPI mAP<sup>val<br>0.5:0.95 | COCOAPI mAP<sup>val<br>0.5 | COCOAPI mAP<sup>test_dev<br>0.5:0.95 | COCOAPI mAP<sup>test_dev<br>0.5 | MatlabAPI mAP<sup>test_dev<br>0.5:0.95 | MatlabAPI mAP<sup>test_dev<br>0.5 | 下载  | 配置文件 |
|:---------|:------:|:------:| :----: | :------:| :------: | :------:| :----: | :------:|
|PP-YOLOE-s|  23.5  |  39.9  |  19.4  |  33.6   |  23.68   |  40.66  | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_s_80e_visdrone.pdparams) | [配置文件](./ppyoloe_crn_s_80e_visdrone.yml) |
|PP-YOLOE-P2-Alpha-s|    24.4  |  41.6  |  20.1  |  34.7  |  24.55   |  42.19  | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_s_p2_alpha_80e_visdrone.pdparams) | [配置文件](./ppyoloe_crn_s_p2_alpha_80e_visdrone.yml) |
|PP_YOLOE_plus_new_s|  25.1  |  42.8  |  20.7  |  36.2   |  25.16  |  43.86   | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_new_crn_s_80e_visdrone.pdparams) | [配置文件](./ppyoloe_plus_new_crn_s_80e_visdrone.yml) |
|PP-YOLOE-l|  29.2  |  47.3  |  23.5  |  39.1   |  28.00   |  46.20  | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_80e_visdrone.pdparams) | [配置文件](./ppyoloe_crn_l_80e_visdrone.yml) |
|PP-YOLOE-P2-Alpha-l|  30.1  |  48.9  |  24.3  |  40.8   |  28.47   |  48.16  | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_p2_alpha_80e_visdrone.pdparams) | [配置文件](./ppyoloe_crn_l_p2_alpha_80e_visdrone.yml) |
|PP_YOLOE_plus_new_l|  31.9  |  52.1  |  25.6  |  43.5   |  30.25  |  51.18   | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_new_crn_l_80e_visdrone.pdparams) | [配置文件](./ppyoloe_plus_new_crn_l_80e_visdrone.yml) |
|PP-YOLOE-Alpha-largesize-l|  41.9  |  65.0 |  32.3  |  53.0   |  37.13   |  61.15  | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_alpha_largesize_80e_visdrone.pdparams) | [配置文件](./ppyoloe_crn_l_alpha_largesize_80e_visdrone.yml) |
|PP-YOLOE-P2-Alpha-largesize-l|  41.3  |  64.5  |  32.4  |  53.1   |  37.49   |  51.54  | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_p2_alpha_largesize_80e_visdrone.pdparams) | [配置文件](./ppyoloe_crn_l_p2_alpha_largesize_80e_visdrone.yml) |
|PP-YOLOE-plus-largesize-l |  43.3  |  66.7 |  33.5  |  54.7   |  38.24   |  62.76  | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_l_largesize_80e_visdrone.pdparams) | [配置文件](./ppyoloe_plus_crn_l_largesize_80e_visdrone.yml) |
|PP-YOLOE-plus_new-largesize_l |  42.7  |  65.9 |  33.6  |  55.1   |  38.4   |  63.07  | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_new_crn_l_largesize_80e_visdrone.pdparams) | [配置文件](./ppyoloe_plus_new_crn_l_largesize_80e_visdrone.yml) |


## 原图评估和拼图评估对比：

|    模型   |       数据集     |  SLICE_SIZE  |  OVERLAP_RATIO  | 类别数  | mAP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | 下载链接  | 配置文件 |
|:---------|:---------------:|:---------------:|:---------------:|:------:|:-----------------------:|:-------------------:|:---------:| :-----: |
|PP-YOLOE-l| VisDrone-DET|  640 | 0.25 | 10 |  29.7 |  48.5 | [下载链接](https://bj.bcebos.com/v1/paddledet/models/ppyoloe_crn_l_80e_sliced_visdrone_640_025.pdparams) | [配置文件](../smalldet/ppyoloe_crn_l_80e_sliced_visdrone_640_025.yml) |
|PP-YOLOE-l (Assembled)| VisDrone-DET|  640 | 0.25 | 10 | 37.2 | 59.4 | [下载链接](https://bj.bcebos.com/v1/paddledet/models/ppyoloe_crn_l_80e_sliced_visdrone_640_025.pdparams) | [配置文件](../smalldet/ppyoloe_crn_l_80e_sliced_visdrone_640_025.yml) |


**注意:**
- PP-YOLOE模型训练过程中使用8 GPUs进行混合精度训练，如果**GPU卡数**或者**batch size**发生了改变，你需要按照公式 **lr<sub>new</sub> = lr<sub>default</sub> * (batch_size<sub>new</sub> * GPU_number<sub>new</sub>) / (batch_size<sub>default</sub> * GPU_number<sub>default</sub>)** 调整学习率。
- 具体使用教程请参考[ppyoloe](../ppyoloe#getting-start)。
- P2表示增加P2层(1/4下采样层)的特征，共输出4个PPYOLOEHead。
- Alpha表示对CSPResNet骨干网络增加可一个学习权重参数Alpha参与训练。
- largesize表示使用以1600尺度为基础的多尺度训练和1920尺度预测，相应的训练batch_size也减小，以速度来换取高精度。
- new 表示使用基于向量的DFL算法和针对小目标的中心先验优化策略，并在模型中加入transformer
- MatlabAPI测试是使用官网评测工具[VisDrone2018-DET-toolkit](https://github.com/VisDrone/VisDrone2018-DET-toolkit)。
- 切图训练模型的配置文件及训练相关流程请参照[smalldet](../smalldet)。


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
