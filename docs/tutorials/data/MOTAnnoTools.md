# 多目标跟踪标注工具



## 目录

* [前期准备](#前期准备)
* [SDE数据集](#SDE数据集)
  * [LabelMe](#LabelMe)
  * [LabelImg](#LabelImg)
* [JDE数据集](#JDE数据集)
  * [DarkLabel](#DarkLabel)


### 前期准备

请先查看[多目标跟踪数据集准备](PrepareMOTDataSet.md)确定MOT模型选型和MOT数据集的类型。
通常综合数据标注成本和模型精度速度平衡考虑，更推荐使用SDE系列数据集，和SDE系列模型的ByteTrack或OC-SORT。SDE系列数据集的标注工具与目标检测任务是一致的。

### SDE数据集
SDE数据集是纯检测标注的数据集，用户自定义数据集可以参照[DET数据准备文档](./PrepareDetDataSet.md)准备。

#### LabelMe
LabelMe的使用可以参考[DetAnnoTools](DetAnnoTools.md)

#### LabelImg
LabelImg的使用可以参考[DetAnnoTools](DetAnnoTools.md)


### JDE数据集
JDE数据集是同时有检测和ReID标注的数据集，标注成本比SDE数据集更高。

#### [DarkLabel](https://github.com/darkpgmr/DarkLabel)



#### 使用说明



#### 标注格式
标注文件需要转化为MOT JDE数据集格式，包含`images`和`labels_with_ids`文件夹，具体参照[用户自定义数据集准备](PrepareMOTDataSet.md#用户自定义数据集准备)。
