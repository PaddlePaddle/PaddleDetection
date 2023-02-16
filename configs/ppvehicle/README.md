简体中文 | [English](README.md)

## PP-YOLOE Vehicle 检测模型

PaddleDetection团队提供了针对自动驾驶场景的基于PP-YOLOE的检测模型，用户可以下载模型进行使用，主要包含5个数据集(BDD100K-DET、BDD100K-MOT、UA-DETRAC、PPVehicle9cls、PPVehicle)。其中前3者为公开数据集，后两者为整合数据集。
- BDD100K-DET具体类别为10类，包括`pedestrian(1), rider(2), car(3), truck(4), bus(5), train(6), motorcycle(7), bicycle(8), traffic light(9), traffic sign(10)`。
- BDD100K-MOT具体类别为8类，包括`pedestrian(1), rider(2), car(3), truck(4), bus(5), train(6), motorcycle(7), bicycle(8)`，但数据集比BDD100K-DET更大更多。
- UA-DETRAC具体类别为4类，包括`car(1), bus(2), van(3), others(4)`。
- PPVehicle9cls数据集整合了BDD100K-MOT和UA-DETRAC，具体类别为9类，包括`pedestrian(1), rider(2), car(3), truck(4), bus(5), van(6), motorcycle(7), bicycle(8), others(9)`。
- PPVehicle数据集整合了BDD100K-MOT和UA-DETRAC，是将BDD100K-MOT中的`car, truck, bus, van`和UA-DETRAC中的`car, bus, van`都合并为1类`vehicle(1)`后的数据集。

相关模型的部署模型均在[PP-Vehicle](../../deploy/pipeline/)项目中使用。

|    模型   |       数据集     | 类别数  | mAP<sup>val<br>0.5:0.95 |  下载链接  | 配置文件 |
|:---------|:---------------:|:------:|:-----------------------:|:---------:| :-----: |
|PP-YOLOE-l|   BDD100K-DET   |   10   |  35.6 | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_36e_bdd100kdet.pdparams) | [配置文件](./ppyoloe_crn_l_36e_bdd100kdet.yml) |
|PP-YOLOE-l|   BDD100K-MOT   |   8    |  33.7 | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_36e_bdd100kmot.pdparams) | [配置文件](./ppyoloe_crn_l_36e_bdd100kmot.yml) |
|PP-YOLOE-l|   UA-DETRAC     |   4    |  51.4 | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_36e_uadetrac.pdparams) | [配置文件](./ppyoloe_crn_l_36e_uadetrac.yml) |
|PP-YOLOE-l|   PPVehicle9cls |   9    |  40.0 | [下载链接](https://paddledet.bj.bcebos.com/models/mot_ppyoloe_l_36e_ppvehicle9cls.pdparams) | [配置文件](./mot_ppyoloe_l_36e_ppvehicle9cls.yml) |
|PP-YOLOE-s|   PPVehicle9cls |   9    |  35.3 | [下载链接](https://paddledet.bj.bcebos.com/models/mot_ppyoloe_s_36e_ppvehicle9cls.pdparams) | [配置文件](./mot_ppyoloe_s_36e_ppvehicle9cls.yml) |
|PP-YOLOE-l|   PPVehicle     |   1    |  63.9 | [下载链接](https://paddledet.bj.bcebos.com/models/mot_ppyoloe_l_36e_ppvehicle.pdparams) | [配置文件](./mot_ppyoloe_l_36e_ppvehicle.yml) |
|PP-YOLOE-s|   PPVehicle     |   1    |  61.3  | [下载链接](https://paddledet.bj.bcebos.com/models/mot_ppyoloe_s_36e_ppvehicle.pdparams) | [配置文件](./mot_ppyoloe_s_36e_ppvehicle.yml) |
|PP-YOLOE+_t-aux(320)|   PPVehicle   |   1    |  53.5  | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_t_auxhead_320_60e_ppvehicle.pdparams) | [配置文件](./ppyoloe_plus_crn_t_auxhead_320_60e_ppvehicle.yml) |


**注意:**
- PP-YOLOE模型训练过程中使用8 GPUs进行混合精度训练，如果**GPU卡数**或者**batch size**发生了改变，你需要按照公式 **lr<sub>new</sub> = lr<sub>default</sub> * (batch_size<sub>new</sub> * GPU_number<sub>new</sub>) / (batch_size<sub>default</sub> * GPU_number<sub>default</sub>)** 调整学习率。
- 具体使用教程请参考[ppyoloe](../ppyoloe#getting-start)。
- 如需预测出对应类别，可自行修改和添加对应的label_list.txt文件(一行记录一个对应种类)，TestDataset中的anno_path为绝对路径，如：
```
TestDataset:
  !ImageFolder
    anno_path: label_list.txt # 如不使用dataset_dir，则anno_path即为相对于PaddleDetection主目录的相对路径
    # dataset_dir: dataset/ppvehicle # 如使用dataset_dir，则dataset_dir/anno_path作为新的anno_path
```
label_list.txt里的一行记录一个对应种类，如下所示：
```
vehicle
```

## YOLOv3 Vehicle 检测模型

请参考[Vehicle_YOLOv3页面](./vehicle_yolov3/README_cn.md)

## PP-OCRv3 车牌识别模型

车牌识别采用Paddle自研超轻量级模型PP-OCRv3_det、PP-OCRv3_rec。在[CCPD数据集](https://github.com/detectRecog/CCPD)（CCPD2019+CCPD2020车牌数据集）上进行了fine-tune。模型训练基于[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/applications/%E8%BD%BB%E9%87%8F%E7%BA%A7%E8%BD%A6%E7%89%8C%E8%AF%86%E5%88%AB.md)完成，我们提供了预测模型下载：

|    模型   |  数据集  | 精度 | 下载  | 配置文件 |
|:---------|:-------:|:------:| :----: | :------:|
| PP-OCRv3_det | CCPD组合数据集 |  hmean:0.979 |[下载链接](https://bj.bcebos.com/v1/paddledet/models/pipeline/ch_PP-OCRv3_det_infer.tar.gz) | [配置文件](https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml) |
| PP-OCRv3_rec | CCPD组合数据集 |  acc:0.773 |[下载链接](https://bj.bcebos.com/v1/paddledet/models/pipeline/ch_PP-OCRv3_rec_infer.tar.gz) | [配置文件](https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/configs/rec/PP-OCRv3/ch_PP-OCRv3_rec_distillation.yml) |

## PP-LCNet 车牌属性模型

车牌属性采用Paddle自研超轻量级模型PP-LCNet。在[VeRi数据集](https://www.v7labs.com/open-datasets/veri-dataset)进行训练。模型训练基于[PaddleClas](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/en/PULC/PULC_vehicle_attribute_en.md)完成，我们提供了预测模型下载：

|    模型   |  数据集  | 精度 | 下载  | 配置文件 |
|:---------|:-------:|:------:| :----: | :------:|
| PP-LCNet_x1_0 | VeRi数据集 |  90.81 |[下载链接](https://bj.bcebos.com/v1/paddledet/models/pipeline/vehicle_attribute_model.zip) | [配置文件](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/ppcls/configs/PULC/vehicle_attribute/PPLCNet_x1_0.yaml) |


## 引用
```
@InProceedings{bdd100k,
    author = {Yu, Fisher and Chen, Haofeng and Wang, Xin and Xian, Wenqi and Chen,
              Yingying and Liu, Fangchen and Madhavan, Vashisht and Darrell, Trevor},
    title = {BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2020}
}

@article{CVIU_UA-DETRAC,
    author = {Longyin Wen and Dawei Du and Zhaowei Cai and Zhen Lei and Ming{-}Ching Chang and
              Honggang Qi and Jongwoo Lim and Ming{-}Hsuan Yang and Siwei Lyu},
    title = {{UA-DETRAC:} {A} New Benchmark and Protocol for Multi-Object Detection and Tracking},
    journal = {Computer Vision and Image Understanding},
    year = {2020}
}  
```
