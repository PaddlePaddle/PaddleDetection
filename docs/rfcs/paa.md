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