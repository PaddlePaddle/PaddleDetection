# PP-YOLOE+ 下游任务

我们验证了PP-YOLOE+模型强大的泛化能力，在农业、低光、工业等不同场景下游任务检测效果稳定提升!

农业数据集采用[Embrapa WGISD](https://github.com/thsant/wgisd)，该数据集用于葡萄栽培中基于图像的监测和现场机器人技术，提供了来自5种不同葡萄品种的实地实例，
处理后的COCO格式，包含图片训练集242张，测试集58张，5个类别，[Embrapa WGISD COCO格式下载](https://bj.bcebos.com/v1/paddledet/data/wgisd.zip)；

低光数据集使用[ExDark](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset/tree/master/Dataset)，该数据集是一个专门在低光照环境下拍摄出针对低光目标检测的数据集，包括从极低光环境到暮光环境等10种不同光照条件下的图片，
处理后的COCO格式，包含图片训练集5891张，测试集1472张，12个类别，[ExDark COCO格式下载](https://bj.bcebos.com/v1/paddledet/data/Exdark.zip)；

工业数据集使用[PKU-Market-PCB](https://robotics.pkusz.edu.cn/resources/dataset/)，该数据集用于印刷电路板（PCB）的瑕疵检测，提供了6种常见的PCB缺陷，
处理后的COCO格式，包含图片训练集555张，测试集138张，6个类别，[PKU-Market-PCB COCO格式下载](https://bj.bcebos.com/v1/paddledet/data/PCB_coco.zip)。

商超数据集[SKU110k](https://github.com/eg4000/SKU110K_CVPR19)是商品超市场景下的密集目标检测数据集，包含11,762张图片和超过170个实例。其中包括8,233张用于训练的图像、588张用于验证的图像和2,941张用于测试的图像。


## 实验结果：

|    模型  |       数据集     | mAP<sup>val<br>0.5:0.95 |  下载链接  | 配置文件 |
|:---------|:---------------:|:-----------------------:|:---------:| :-----: |
|PP-YOLOE_m|   Embrapa WGISD  |  52.7 | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_m_80e_wgisd.pdparams) | [配置文件](./ppyoloe_crn_m_80e_wgisd.yml) |
|PP-YOLOE+_m<br>(obj365_pretrained)|   Embrapa WGISD  |  60.8(+8.1) | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_m_80e_obj365_pretrained_wgisd.pdparams) | [配置文件](./ppyoloe_plus_crn_m_80e_obj365_pretrained_wgisd.yml) |
|PP-YOLOE+_m<br>(coco_pretrained)|   Embrapa WGISD  |  59.7(+7.0) | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_m_80e_coco_pretrained_wgisd.pdparams) | [配置文件](./ppyoloe_plus_crn_m_80e_coco_pretrained_wgisd.yml) |
|PP-YOLOE_m|      ExDark      |  56.4 | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_m_80e_exdark.pdparams) | [配置文件](./ppyoloe_crn_m_80e_exdark.yml) |
|PP-YOLOE+_m<br>(obj365_pretrained)|   ExDark  |  57.7(+1.3) | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_m_80e_obj365_pretrained_exdark.pdparams) | [配置文件](./ppyoloe_plus_crn_m_80e_obj365_pretrained_exdark.yml) |
|PP-YOLOE+_m<br>(coco_pretrained)|   ExDark  |  58.1(+1.7) | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_m_80e_coco_pretrained_exdark.pdparams) | [配置文件](./ppyoloe_plus_crn_m_80e_coco_pretrained_exdark.yml) |
|PP-YOLOE_m|      PKU-Market-PCB      |  50.8 | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_m_80e_pcb.pdparams) | [配置文件](./ppyoloe_crn_m_80e_pcb.yml) |
|PP-YOLOE+_m<br>(obj365_pretrained)|   PKU-Market-PCB  |  52.7(+1.9) | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_m_80e_obj365_pretrained_pcb.pdparams) | [配置文件](./ppyoloe_plus_crn_m_80e_obj365_pretrained_pcb.yml) |
|PP-YOLOE+_m<br>(coco_pretrained)|   PKU-Market-PCB  |  52.4(+1.6) | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_m_80e_coco_pretrained_pcb.pdparams) | [配置文件](./ppyoloe_plus_crn_m_80e_coco_pretrained_pcb.yml) |

**注意:**
- PP-YOLOE模型训练过程中使用8 GPUs进行训练，如果**GPU卡数**或者**batch size**发生了改变，你需要按照公式 **lr<sub>new</sub> = lr<sub>default</sub> * (batch_size<sub>new</sub> * GPU_number<sub>new</sub>) / (batch_size<sub>default</sub> * GPU_number<sub>default</sub>)** 调整学习率。
- 具体使用教程请参考[ppyoloe](../ppyoloe#getting-start)。  


## SKU110k Model ZOO
|     Model      | Epoch | GPU number | images/GPU |  backbone  | input shape | Box AP<sup>val<br>0.5:0.95 (maxDets=300) | Box AP<sup>test<br>0.5:0.95 (maxDets=300) | download | config |
|:--------------:|:-----:|:-------:|:----------:|:----------:| :-------:|:-------------------------:|:---------------------------:|:---------:|:------:|
| PP-YOLOE+_s | 80 | 8 | 8 | cspresnet-s | 960 | 57.4 | 58.8 | [download](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_s_80e_sku110k.pdparams) | [config](./ppyoloe_plus_crn_s_80e_sku110k.yml) |
| PP-YOLOE+_m | 80 | 8 | 8 | cspresnet-m | 960 | 58.2 | 59.7 | [download](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_m_80e_sku110k.pdparams) | [config](./ppyoloe_plus_crn_m_80e_sku110k.yml) |
| PP-YOLOE+_l | 80 | 8 | 4 | cspresnet-l | 960 | 58.8 | 60.2 | [download](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_l_80e_sku110k.pdparams) | [config](./ppyoloe_plus_crn_l_80e_sku110k.yml) |
| PP-YOLOE+_x | 80 | 8 | 4 | cspresnet-x | 960 | 59.0 | 60.3 | [download](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_x_80e_sku110k.pdparams) | [config](./ppyoloe_plus_crn_x_80e_sku110k.yml) |


**注意:**
- SKU110k系列模型训练过程中使用8 GPUs进行训练，如果**GPU卡数**或者**batch size**发生了改变，你需要按照公式 **lr<sub>new</sub> = lr<sub>default</sub> * (batch_size<sub>new</sub> * GPU_number<sub>new</sub>) / (batch_size<sub>default</sub> * GPU_number<sub>default</sub>)** 调整学习率。
- SKU110k数据集使用**maxDets=300**的mAP值作为评估指标。
- 具体使用教程请参考[ppyoloe](../ppyoloe#getting-start)。


## 引用
```
@inproceedings{goldman2019dense,
 author    = {Eran Goldman and Roei Herzig and Aviv Eisenschtat and Jacob Goldberger and Tal Hassner},
 title     = {Precise Detection in Densely Packed Scenes},
 booktitle = {Proc. Conf. Comput. Vision Pattern Recognition (CVPR)},
 year      = {2019}
}

@article{Exdark,
title={Getting to Know Low-light Images with The Exclusively Dark Dataset},
author={Loh, Yuen Peng and Chan, Chee Seng},
journal={Computer Vision and Image Understanding},
volume={178},
pages={30-42},
year={2019},
doi={https://doi.org/10.1016/j.cviu.2018.10.010}
}
```
