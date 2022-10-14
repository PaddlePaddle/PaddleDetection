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
| GFL_ResNet18-vd + LD   | student          | 38.2  | [model](https://bj.bcebos.com/v1/paddledet/models/gfl_ld_r18vd_1x_coco.pdparams), [config1](../../gfl/gfl_ld_r18vd_1x_coco.yml), [config2](./gfl_ld_distill.yml) |

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
```
