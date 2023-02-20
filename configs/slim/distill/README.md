# Distillation(蒸馏)

## YOLOv3模型蒸馏
以YOLOv3-MobileNetV1为例，使用YOLOv3-ResNet34作为蒸馏训练的teacher网络, 对YOLOv3-MobileNetV1结构的student网络进行蒸馏。
COCO数据集作为目标检测任务的训练目标难度更大，意味着teacher网络会预测出更多的背景bbox，如果直接用teacher的预测输出作为student学习的`soft label`会有严重的类别不均衡问题。解决这个问题需要引入新的方法，详细背景请参考论文:[Object detection at 200 Frames Per Second](https://arxiv.org/abs/1805.06361)。
为了确定蒸馏的对象，我们首先需要找到student和teacher网络得到的`x,y,w,h,cls,objness`等Tensor，用teacher得到的结果指导student训练。具体实现可参考[代码](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/ppdet/slim/distill.py)


## FGD模型蒸馏

FGD全称为[Focal and Global Knowledge Distillation for Detectors](https://arxiv.org/abs/2111.11837v1)，是目标检测任务的一种蒸馏方法，FGD蒸馏分为两个部分`Focal`和`Global`。`Focal`蒸馏分离图像的前景和背景，让学生模型分别关注教师模型的前景和背景部分特征的关键像素；`Global`蒸馏部分重建不同像素之间的关系并将其从教师转移到学生，以补偿`Focal`蒸馏中丢失的全局信息。试验结果表明，FGD蒸馏算法在基于anchor和anchor free的方法上能有效提升模型精度。
在PaddleDetection中，我们实现了FGD算法，并基于retinaNet算法进行验证，实验结果如下：
| algorithm | model | AP | download|
|:-:| :-: | :-: | :-:|
|retinaNet_r101_fpn_2x | teacher | 40.6 | [download](https://paddledet.bj.bcebos.com/models/retinanet_r101_fpn_2x_coco.pdparams) |
|retinaNet_r50_fpn_1x| student | 37.5 |[download](https://paddledet.bj.bcebos.com/models/retinanet_r50_fpn_1x_coco.pdparams) |
|retinaNet_r50_fpn_2x + FGD| student | 40.8 |[download](https://paddledet.bj.bcebos.com/models/retinanet_r101_distill_r50_2x_coco.pdparams) |


## LD模型蒸馏

LD全称为[Localization Distillation for Dense Object Detection](https://arxiv.org/abs/2102.12252)，将回归框表示为概率分布，把分类任务的KD用在定位任务上，并且使用因地制宜、分而治之的策略，在不同的区域分别学习分类知识与定位知识。在PaddleDetection中，我们实现了LD算法，并基于GFL模型进行验证，实验结果如下：
| algorithm | model | AP | download|
|:-:| :-: | :-: | :-:|
| GFL_ResNet101-vd   | teacher          | 46.8  | [model](https://paddledet.bj.bcebos.com/models/gfl_r101vd_fpn_mstrain_2x_coco.pdparams), [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/gfl/gfl_r101vd_fpn_mstrain_2x_coco.yml) |
| GFL_ResNet18-vd   | student          | 36.6  | [model](https://paddledet.bj.bcebos.com/models/gfl_r18vd_1x_coco.pdparams), [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/gfl/gfl_r18vd_1x_coco.yml) |
| GFL_ResNet18-vd + LD   | student          | 38.2  | [model](https://bj.bcebos.com/v1/paddledet/models/gfl_slim_ld_r18vd_1x_coco.pdparams), [config1](../../gfl/gfl_slim_ld_r18vd_1x_coco.yml), [config2](./gfl_ld_distill.yml) |

## CWD模型蒸馏

CWD全称为[Channel-wise Knowledge Distillation for Dense Prediction*](https://arxiv.org/pdf/2011.13256.pdf)，通过最小化教师网络与学生网络的通道概率图之间的 Kullback-Leibler (KL) 散度，使得在蒸馏过程更加关注每个通道的最显著的区域，进而提升文本检测与图像分割任务的精度。在PaddleDetection中，我们实现了CWD算法，并基于GFL和RetinaNet模型进行验证，实验结果如下：
| algorithm | model | AP | download|
|:-:| :-: | :-: | :-:|
|retinaNet_r101_fpn_2x | teacher | 40.6 | [download](https://paddledet.bj.bcebos.com/models/retinanet_r101_fpn_2x_coco.pdparams) |
|retinaNet_r50_fpn_1x| student | 37.5 |[download](https://paddledet.bj.bcebos.com/models/retinanet_r50_fpn_1x_coco.pdparams) |
|retinaNet_r50_fpn_2x + CWD| student | 40.5 |[download](https://paddledet.bj.bcebos.com/models/retinanet_r50_fpn_2x_coco_cwd.pdparams) |
|gfl_r101_fpn_2x | teacher | 46.8 | [download](https://paddledet.bj.bcebos.com/models/gfl_r101vd_fpn_mstrain_2x_coco.pdparams) |
|gfl_r50_fpn_1x| student | 41.0 |[download](https://paddledet.bj.bcebos.com/models/gfl_r50_fpn_1x_coco.pdparams) |
|gfl_r50_fpn_2x + CWD| student | 44.0 |[download](https://paddledet.bj.bcebos.com/models/gfl_r50_fpn_2x_coco_cwd.pdparams) |

## PPYOLOE+模型蒸馏



## 快速开始

### 训练
```shell
# 单卡
python tools/train.py -c configs/ppyoloe/distill/ppyoloe_plus_crn_l_80e_coco_distill.yml --slim_config configs/slim/distill/ppyoloe_plus_distill_x_to_l.yml
# 多卡
python3.7 -m paddle.distributed.launch --log_dir=ppyoloe_plus_distill_x_to_l/ --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/ppyoloe/distill/ppyoloe_plus_crn_l_80e_coco_distill.yml --slim_config configs/slim/distill/ppyoloe_plus_distill_x_to_l.yml
```

- `-c`: 指定模型配置文件，也是student配置文件。
- `--slim_config`: 指定压缩策略配置文件，也是teacher配置文件。

### 评估
```shell
python tools/eval.py -c configs/ppyoloe/distill/ppyoloe_plus_crn_l_80e_coco_distill.yml -o weights=output/ppyoloe_plus_crn_l_80e_coco_distill/model_final.pdparams
```

- `-c`: 指定模型配置文件，也是student配置文件。
- `--slim_config`: 指定压缩策略配置文件，也是teacher配置文件。
- `-o weights`: 指定压缩算法训好的模型路径。

### 测试
```shell
python tools/infer.py -c configs/ppyoloe/distill/ppyoloe_plus_crn_l_80e_coco_distill.yml -o weights=output/ppyoloe_plus_crn_l_80e_coco_distill/model_final.pdparams --infer_img=demo/000000014439_640x640.jpg
```

- `-c`: 指定模型配置文件。
- `--slim_config`: 指定压缩策略配置文件。
- `-o weights`: 指定压缩算法训好的模型路径。
- `--infer_img`: 指定测试图像路径。


## Citations
```
@article{mehta2018object,
      title={Object detection at 200 Frames Per Second},
      author={Rakesh Mehta and Cemalettin Ozturk},
      year={2018},
      eprint={1805.06361},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@inproceedings{yang2022focal,
  title={Focal and global knowledge distillation for detectors},
  author={Yang, Zhendong and Li, Zhe and Jiang, Xiaohu and Gong, Yuan and Yuan, Zehuan and Zhao, Danpei and Yuan, Chun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4643--4652},
  year={2022}
}

@Inproceedings{zheng2022LD,
  title={Localization Distillation for Dense Object Detection},
  author= {Zheng, Zhaohui and Ye, Rongguang and Wang, Ping and Ren, Dongwei and Zuo, Wangmeng and Hou, Qibin and Cheng, Mingming},
  booktitle={CVPR},
  year={2022}
}

@inproceedings{shu2021channel,
  title={Channel-wise knowledge distillation for dense prediction},
  author={Shu, Changyong and Liu, Yifan and Gao, Jianfei and Yan, Zheng and Shen, Chunhua},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={5311--5320},
  year={2021}
}
```
