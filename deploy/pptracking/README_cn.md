[English](README_en.md) | 简体中文

# 实时多目标跟踪系统PP-Tracking

PP-Tracking是基于PaddlePaddle深度学习框架的业界首个开源的实时多目标跟踪系统，具有模型丰富、应用广泛和部署高效三大优势。
PP-Tracking支持单镜头跟踪(MOT)和跨镜头跟踪(MTMCT)两种模式，针对实际业务的难点和痛点，提供了行人跟踪、车辆跟踪、多类别跟踪、小目标跟踪、流量统计以及跨镜头跟踪等各种多目标跟踪功能和应用，部署方式支持API调用和GUI可视化界面，部署语言支持Python和C++，部署平台环境支持Linux、NVIDIA Jetson等。

<div width="1000" align="center">
  <img src="../../docs/images/pptracking.png"/>
</div>

<div width="1000" align="center">
  <img src="../../docs/images/pptracking-demo.gif"/>
  <br>
  视频来源：VisDrone和BDD100K公开数据集</div>
</div>


## 一、快速开始

### AI Studio公开项目案例
PP-Tracking 提供了AI Studio公开项目案例，教程请参考[PP-Tracking之手把手玩转多目标跟踪](https://aistudio.baidu.com/aistudio/projectdetail/3022582)。

### Python端预测部署
PP-Tracking 支持Python预测部署，教程请参考[PP-Tracking Python部署文档](python/README.md)。

### C++端预测部署
PP-Tracking 支持C++预测部署，教程请参考[PP-Tracking C++部署文档](cpp/README.md)。

### GUI可视化界面预测部署
PP-Tracking 提供了简洁的GUI可视化界面，教程请参考[PP-Tracking可视化界面试用版使用文档](https://github.com/yangyudong2020/PP-Tracking_GUi)。


## 二、算法介绍

PP-Tracking 支持单镜头跟踪(MOT)和跨镜头跟踪(MTMCT)两种模式。
- 单镜头跟踪同时支持**FairMOT**和**DeepSORT**两种多目标跟踪算法，跨镜头跟踪只支持**DeepSORT**算法。
- 单镜头跟踪的功能包括行人跟踪、车辆跟踪、多类别跟踪、小目标跟踪以及流量统计，模型主要是基于FairMOT进行优化，实现了实时跟踪的效果，同时基于不同应用场景提供了针对性的预训练模型。
- DeepSORT算法方案(包括跨镜头跟踪用到的DeepSORT)，选用的检测器是PaddleDetection自研的高性能检测模型[PP-YOLOv2](../../ppyolo/)和轻量级特色检测模型[PP-PicoDet](../../picodet/)，选用的ReID模型是PaddleClas自研的超轻量骨干网络模型[PP-LCNet](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/docs/zh_CN/models/PP-LCNet.md)

PP-Tracking中提供的多场景预训练模型以及导出后的预测部署模型如下：

| 场景            | 数据集               | 精度（MOTA） | 预测速度（FPS） | 配置文件 | 模型权重 | 预测部署模型 |
| :---------:     |:---------------     | :-------:  | :------:      | :------:|:-----: | :--------: |
| 行人跟踪         | MOT17               | 65.3       | 23.9           | [配置文件](../../configs/mot/fairmot/fairmot_hrnetv2_w18_dlafpn_30e_576x320.yml) | [下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_hrnetv2_w18_dlafpn_30e_576x320.pdparams) | [下载链接](https://bj.bcebos.com/v1/paddledet/models/mot/fairmot_hrnetv2_w18_dlafpn_30e_576x320.tar) |
| 行人小目标跟踪    | VisDrone-pedestrian |  40.5       | 8.35          | [配置文件](../../configs/mot/pedestrian/fairmot_hrnetv2_w18_dlafpn_30e_1088x608_visdrone_pedestrian.yml) | [下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_hrnetv2_w18_dlafpn_30e_1088x608_visdrone_pedestrian.pdparams) | [下载链接](https://bj.bcebos.com/v1/paddledet/models/mot/fairmot_hrnetv2_w18_dlafpn_30e_1088x608_visdrone_pedestrian.tar) |
| 车辆跟踪         | BDD100k-vehicle    | 32.6         | 24.3          | [配置文件](../../configs/mot/vehicle/fairmot_hrnetv2_w18_dlafpn_30e_576x320_bdd100kmot_vehicle.yml) | [下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_hrnetv2_w18_dlafpn_30e_576x320_bdd100kmot_vehicle.pdparams) | [下载链接](https://bj.bcebos.com/v1/paddledet/models/mot/fairmot_hrnetv2_w18_dlafpn_30e_576x320_bdd100kmot_vehicle.tar) |
| 车辆小目标跟踪    | VisDrone-vehicle   | 39.8         | 22.8          | [配置文件](../../configs/mot/vehicle/fairmot_hrnetv2_w18_dlafpn_30e_576x320_visdrone_vehicle.yml) | [下载链接](https://paddledet.bj.bcebos.com/models/mot/fairmot_hrnetv2_w18_dlafpn_30e_576x320_visdrone_vehicle.pdparams) | [下载链接](https://bj.bcebos.com/v1/paddledet/models/mot/fairmot_hrnetv2_w18_dlafpn_30e_576x320_visdrone_vehicle.tar)
| 多类别跟踪       | BDD100k             |  -          | 12.5          | [配置文件](../../configs/mot/mcfairmot/mcfairmot_hrnetv2_w18_dlafpn_30e_576x320_bdd100k_mcmot.yml) | [下载链接](https://paddledet.bj.bcebos.com/models/mot/mcfairmot_hrnetv2_w18_dlafpn_30e_576x320_bdd100k_mcmot.pdparams) | [下载链接](https://bj.bcebos.com/v1/paddledet/models/mot/mcfairmot_hrnetv2_w18_dlafpn_30e_576x320_bdd100k_mcmot.tar) |
| 多类别小目标跟踪  | VisDrone            |  20.4       | 6.74          | [配置文件](../../configs/mot/mcfairmot/mcfairmot_hrnetv2_w18_dlafpn_30e_1088x608_visdrone.yml) | [下载链接](https://paddledet.bj.bcebos.com/models/mot/mcfairmot_hrnetv2_w18_dlafpn_30e_1088x608_visdrone.pdparams) | [下载链接](https://bj.bcebos.com/v1/paddledet/models/mot/mcfairmot_hrnetv2_w18_dlafpn_30e_1088x608_visdrone.tar) |

**注意：**
1. 模型预测速度的设备为**NVIDIA Jetson Xavier NX**，速度为**TensorRT FP16**速度，测试环境为CUDA 10.2、JETPACK 4.5.1、TensorRT 7.1。
2. 模型权重是指使用PaddleDetection训练完直接保存的权重，更多跟踪模型权重请参考[多目标跟踪模型库](../../configs/mot/README.md#模型库)去下载，也可按照相应模型配置文件去训练。
3. 预测部署模型是指导出后的前向参数的模型，因为PP-Tracking项目的部署过程中只需要前向参数，可根据[多目标跟踪模型库](../../configs/mot/README.md#模型库)去下载并导出，也可按照相应模型配置文件去训练并导出。导出后的模型文件夹应包括`infer_cfg.yml`、`model.pdiparams`、`model.pdiparams.info`和`model.pdmodel`四个文件，一般会将它们以tar格式打包。


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
@InProceedings{bdd100k,
    author = {Yu, Fisher and Chen, Haofeng and Wang, Xin and Xian, Wenqi and Chen,
              Yingying and Liu, Fangchen and Madhavan, Vashisht and Darrell, Trevor},
    title = {BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2020}
}
@article{zhang2020fair,
  title={FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking},
  author={Zhang, Yifu and Wang, Chunyu and Wang, Xinggang and Zeng, Wenjun and Liu, Wenyu},
  journal={arXiv preprint arXiv:2004.01888},
  year={2020}
}
@inproceedings{Wojke2018deep,
  title={Deep Cosine Metric Learning for Person Re-identification},
  author={Wojke, Nicolai and Bewley, Alex},
  booktitle={2018 IEEE Winter Conference on Applications of Computer Vision (WACV)},
  year={2018},
  pages={748--756},
  organization={IEEE},
  doi={10.1109/WACV.2018.00087}
}
```
