# 多目标跟踪标注工具



## 目录

* [前期准备](#前期准备)
* [SDE数据集](#SDE数据集)
  * [LabelMe](#LabelMe)
  * [LabelImg](#LabelImg)
* [JDE数据集](#JDE数据集)
  * [DarkLabel](#DarkLabel)
  * [标注格式](#标注格式)


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

##### 安装

从官方给出的下载[链接](https://github.com/darkpgmr/DarkLabel/releases)中下载想要的版本，Windows环境解压后能够直接使用

**视频/图片标注过程**

1. 启动应用程序后，能看到左侧的工具栏
2. 选择视频/图像文件后，按需选择标注形式：
   * Box仅绘制标注框
   * Box+Label绘制标注框&标签
   * Box+Label+AutoID绘制标注框&标签&ID号
   * Popup LabelSelect可以自行定义标签
3. 在视频帧/图像上进行拖动鼠标，进行标注框的绘制
4. 绘制完成后，在上数第六行里选择保存标注文件的形式，默认.txt

![1](https://user-images.githubusercontent.com/34162360/179673519-511b4167-97ed-4228-8869-db9c69a68b6b.mov)



##### 注意事项

1. 如果标注的是视频文件，需要在工具栏上数第五行的下拉框里选择`[fn,cname,id,x1,y1,w,h]` （DarkLabel2.4版本）
2. 鼠标移动到标注框所在区域，右键可以删除标注框
3. 按下shift，可以选中标注框，进行框的移动和对某条边的编辑
4. 按住enter回车，可以自动跟踪标注目标
5. 自动跟踪标注目标过程中可以暂停（松开enter），按需修改标注框



##### 其他使用参考视频

* [DarkLabel (Video/Image Annotation Tool) - Ver.2.0](https://www.youtube.com/watch?v=lok30aIZgUw) 
* [DarkLabel (Image/Video Annotation Tool)](https://www.youtube.com/watch?v=vbydG78Al8s&t=11s)



#### 标注格式
标注文件需要转化为MOT JDE数据集格式，包含`images`和`labels_with_ids`文件夹，具体参照[用户自定义数据集准备](PrepareMOTDataSet.md#用户自定义数据集准备)。
