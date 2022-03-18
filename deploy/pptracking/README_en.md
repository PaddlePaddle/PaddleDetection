English | [简体中文](README_cn.md)

# Real-time Multi-Object Tracking System--PP-Tracking

PP-Tracking is the first open-source real-time Multi-Object Tracking system, and it is based on PaddlePaddle's deep learning framework. It has rich models, wide applications and efficient deployment.

PP-Tracking supports two paradigms: multi-object tracking (MOT) and multi-camera tracking (MTMCT). Considering practical difficulties, PP-Tracking provides various MOT solutions such as pedestrian tracking, vehicle tracking, multi-class tracking, small object tracking, traffic statistics and multi-camera tracking. Available deployment methods include API and GUI visual interface, the available languages include Python and C++, and options the platform environment are Linux, NVIDIA Jetson, etc.

<div width="1000" align="center">
  <img src="../../docs/images/pptracking_en.png"/>
</div>

<div width="1000" align="center">
  <img src="../../docs/images/pptracking-demo.gif"/>
  <br>
  video source：VisDrone and BDD100K datasets</div>
</div>


## I. Quick Start

### AI studio public projects
PP-tracking provides AI studio public projects. You can refer to this [tutorial](https://aistudio.baidu.com/aistudio/projectdetail/3022582).

### Inference and deployment using Python
PP-Tracking supports Python predict and deployment. Please refer to this [doc](python/README.md).

### Inference and deployment using C++
PP-Tracking is compatible with inference and deployment using C++. Please refer to this [doc](cpp/README.md).

### GUI inference and deployment
PP-Tracking supports GUI inference and deployment. Please refer to this [doc](https://github.com/yangyudong2020/PP-Tracking_GUi).

（thanks@[yangyudong2020](https://github.com/yangyudong2020)、@[hchhtc123](https://github.com/hchhtc123) for the contribution to PaddleDetetcion.）

## II. Model Zoo

PP-Tracking has two paradigms: multi-object tracking (MOT) and multi-target multi-camera tracking (MTMCT).
- Multi-object tracking has two models: **FairMOT** and **DeepSORT**, MTMCT only support **DeepSORT**.
- The applications of multi-object tracking include pedestrian tracking, vehicle tracking, multi-class tracking, small object tracking and traffic statistics. The models are mainly optimized based on FairMOT to achieve real-time tracking. At the same time, PP-Tracking provides pre-trained models considering different application scenarios.
- In DeepSORT (including DeepSORT used in MTMCT), the selected detectors are [PP-YOLOv2](../../configs/ppyolo/) with high peformance and the lightweight detector [PP-PicoDet](../../configs/picodet/), both of which are developed by PaddeDetection. And the selected ReID model is PaddleClas's self-developed ultra lightweight backbone [PP-LCNet](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/docs/zh_CN/models/PP-LCNet.md)

Pre-trainied models for multiple scenarios offered by PP-Tracking and the exported deployment models:

| Scenario            | Dataset                | Precision (MOTA)       | Speed (FPS) | Config | Model Weight  | Inference and Deployment Model |
| :---------:      |:---------------        | :-------:  | :------:    | :------:|:-----: | :------------:  |
| pedestrian tracking     | MOT17                  | 65.3       | 23.9        | [config](../../configs/mot/fairmot/fairmot_hrnetv2_w18_dlafpn_30e_576x320.yml) | [download](https://paddledet.bj.bcebos.com/models/mot/fairmot_hrnetv2_w18_dlafpn_30e_576x320.pdparams) | [download](https://bj.bcebos.com/v1/paddledet/models/mot/fairmot_hrnetv2_w18_dlafpn_30e_576x320.tar) |
| pedestrian tracking (small objects) | VisDrone-pedestrian |  40.5| 8.35        | [config](../../configs/mot/pedestrian/fairmot_hrnetv2_w18_dlafpn_30e_1088x608_visdrone_pedestrian.yml) | [download](https://paddledet.bj.bcebos.com/models/mot/fairmot_hrnetv2_w18_dlafpn_30e_1088x608_visdrone_pedestrian.pdparams) | [download](https://bj.bcebos.com/v1/paddledet/models/mot/fairmot_hrnetv2_w18_dlafpn_30e_1088x608_visdrone_pedestrian.tar) |
| vehicle tracking       | BDD100k-vehicle    | 32.6           | 24.3         | [config](../../configs/mot/vehicle/fairmot_hrnetv2_w18_dlafpn_30e_576x320_bdd100kmot_vehicle.yml) | [download](https://paddledet.bj.bcebos.com/models/mot/fairmot_hrnetv2_w18_dlafpn_30e_576x320_bdd100kmot_vehicle.pdparams)| [download](https://bj.bcebos.com/v1/paddledet/models/mot/fairmot_hrnetv2_w18_dlafpn_30e_576x320_bdd100kmot_vehicle.tar) |
| vehicle tracking (small objects)| VisDrone-vehicle   | 39.8      | 22.8        | [config](../../configs/mot/vehicle/fairmot_hrnetv2_w18_dlafpn_30e_576x320_visdrone_vehicle.yml) | [download](https://paddledet.bj.bcebos.com/models/mot/fairmot_hrnetv2_w18_dlafpn_30e_576x320_visdrone_vehicle.pdparams) | [download](https://bj.bcebos.com/v1/paddledet/models/mot/fairmot_hrnetv2_w18_dlafpn_30e_576x320_visdrone_vehicle.tar)
| multi-class tracking      | BDD100k                |  -        | 12.5        | [config](../../configs/mot/mcfairmot/mcfairmot_hrnetv2_w18_dlafpn_30e_576x320_bdd100k_mcmot.yml) | [download](https://paddledet.bj.bcebos.com/models/mot/mcfairmot_hrnetv2_w18_dlafpn_30e_576x320_bdd100k_mcmot.pdparams) | [download](https://bj.bcebos.com/v1/paddledet/models/mot/mcfairmot_hrnetv2_w18_dlafpn_30e_576x320_bdd100k_mcmot.tar) |
| multi-class tracking (small objects)  | VisDrone     |  20.4     | 6.74        | [config](../../configs/mot/mcfairmot/mcfairmot_hrnetv2_w18_dlafpn_30e_1088x608_visdrone.yml) | [download](https://paddledet.bj.bcebos.com/models/mot/mcfairmot_hrnetv2_w18_dlafpn_30e_1088x608_visdrone.pdparams) | [download](https://bj.bcebos.com/v1/paddledet/models/mot/mcfairmot_hrnetv2_w18_dlafpn_30e_1088x608_visdrone.tar) |

**Note：**
1. The equipment for model inference is **NVIDIA Jetson Xavier NX**, the speed is that of **TensorRT FP16**, and the test environment includes CUDA 10.2, JETPACK 4.5.1, and TensorRT 7.1.
2. `Model Weight` means the weight saved after PaddleDetection training. For more about the weight of tracking models, please refer to [modelzoo](../../configs/mot/README.md#模型库). You can also train models according to the corresponding config files and get the model weight.
3. `Inference and Deployment Model` means the exported models with forward parameters, because only forward parameters are required in the deployment of the PP-Tracking project. It can be downloaded and exported according to [modelzoo](../../configs/mot/README.md#模型库), amd you can also train according to the corresponding model config file and get the model weights, and then export them. In exported model files, there should be four files: `infer_cfg.yml`,`model.pdiparams`,`model.pdiparams.info` and `model.pdmodel`, which are generally packaged in the format of tar.


## Citations
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
