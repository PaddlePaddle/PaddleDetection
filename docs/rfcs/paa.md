功能名称： 为 PaddleDetection 添加PAA模型

GitHub Issue：[PaddleDetection#4219](https://github.com/PaddlePaddle/PaddleDetection/issues/4219)

# 概述
通过GMM来拟合anchor score的分布，再根据该GMM分布来区分anchor为正例或负例，来进行anchor的分配

# 动机
不需要人为的设置anchor正例负例与gt box的iou阈值，简化训练参数的设计并提高准确率

# 流程

- 首先根据featmap的尺寸构建anchors grid
- 根据与gt box的iou阈值，第一次匹配anchor与gt box得到候选anchor集合
- 根据上一步获得的候选anchor集合，计算anchor score: 论文的公式3
- 在每个特征尺度级别上选择K个最高anchor score的anchor
- 通过GMM拟合上一步所有特征尺度选择出的anchor score的分布
- 使用该GMM来重新划分anchor正例，负例
- 根据anchor正负例来进行bbox分类(objectness, class)的反向传播, bbox的回归以及iou分支的预测

# 功能列表

- 构建单阶段目标检测器 (与mmdetection的实现相同)
- 根据与每个gt box的iou来获取候选的anchor集合
- 为上一步获得的候选anchor集合，计算anchor score: 论文的公式3
- 在每个特征尺度级别上选择K个最高anchor score的anchor
- GMM拟合分布及根据分布划分样本
- 额外的IOU预测分支


# 概要设计

- 构建单阶段目标检测器类PAA

PAA包括backbone, 检测头

- 构建backbone

基于resnet构建backbone

- 构建检测头PAAHead

包括anchor generator根据backbone生成的featmap的尺寸构建anchors grid

根据每个gt box与anchors的iou阈值，匹配anchor与gt box得到候选anchor集合

根据正负例数量比例来采样正负例anchors (第一次匹配anchor)

对第一次匹配的anchor计算anchor score: bbox的分类loss+回归loss

遍历每个gt box:
  在每个特征尺度上选择前k个anchor score最高的加入anchor样本集合
  对所有尺度的anchor score样本集合进行GMM拟合
  根据拟合的GMM对anchor进行新的正负例划分

对最终划分的anchor进行训练

- 新的IOU预测分支

构建PAAFPN, 用于添加新的IOU预测分支. 因为测试时没有gt box, 也就无法计算gt box与bbox的iou, 无法继续进行后续的计算, 所以使用该IOU预测分支来预测IOU.

# 详细设计

- 构建单阶段目标检测器类PAA

```
class PAA(BaseArch):
  def __init__:
    # build backbone
    # build PAAHead

  # build detector from config file
  def from_config

  # calculate loss (mainly for traning)
  def get_loss

  # calculate prediction (mainly for inference)
  def get_pred
```

- 检测头PAAHead

```
class PAAHead(nn.Layer):
  # 生成anchors
  def get_anchors()

  # 根据每个gt box与anchors的iou阈值，匹配anchor与gt box, 并标记正负例
  def label_box()

  # 根据正负例数量比例来采样正负例anchors
  def subsample_boxes()

  # 计算anchor的anchor score
  def get_anchor_score()

  # 使用anchor score拟合的GMM，重新划分anchor的正负例
  def paa_reassign():
    #在每个特征尺度上选择前k个anchor score最高的加入anchor样本集合
    
    #对所有尺度的anchor score样本集合进行GMM拟合 (使用sklearn的GaussianMixture)
    
    #根据拟合的GMM对anchor进行新的正负例划分

  # 使用GMM划分anchor正负例
  def gmm_separation()
```

- 新的IOU预测分支
构建PAAFPN, 用于添加新的IOU预测分支

```
class PAAFPN(nn.Layer):
    def forward
```

# 训练和预测方法

通过配置文件指定对应参数，PAAHead项指定PAAHead的参数

`tools/train.py`

训练

`tools/eval.py`

评估

`tools/infer.py`

测试

