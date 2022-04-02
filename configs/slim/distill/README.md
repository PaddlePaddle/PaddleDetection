# Distillation(蒸馏)

## YOLOv3模型蒸馏
以YOLOv3-MobileNetV1为例，使用YOLOv3-ResNet34作为蒸馏训练的teacher网络, 对YOLOv3-MobileNetV1结构的student网络进行蒸馏。
COCO数据集作为目标检测任务的训练目标难度更大，意味着teacher网络会预测出更多的背景bbox，如果直接用teacher的预测输出作为student学习的`soft label`会有严重的类别不均衡问题。解决这个问题需要引入新的方法，详细背景请参考论文:[Object detection at 200 Frames Per Second](https://arxiv.org/abs/1805.06361)。
为了确定蒸馏的对象，我们首先需要找到student和teacher网络得到的`x,y,w,h,cls,objness`等Tensor，用teacher得到的结果指导student训练。具体实现可参考[代码](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/ppdet/slim/distill.py)

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
```
