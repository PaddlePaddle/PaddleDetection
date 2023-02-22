简体中文 | [English](PrepareKeypointDataSet_en.md)

# 关键点数据准备
## 目录
- [COCO数据集](#COCO数据集)
- [MPII数据集](#MPII数据集)
- [用户数据准备](#用户数据准备)
    - [数据格式转换](#数据格式转换)
    - [自定义数据训练](#自定义数据训练)

## COCO数据集
### COCO数据集的准备
我们提供了一键脚本来自动完成COCO2017数据集的下载及准备工作，请参考[COCO数据集下载](https://github.com/PaddlePaddle/PaddleDetection/blob/f0a30f3ba6095ebfdc8fffb6d02766406afc438a/docs/tutorials/PrepareDetDataSet.md#COCO%E6%95%B0%E6%8D%AE)。

### COCO数据集（KeyPoint）说明
在COCO中，关键点序号与部位的对应关系为：
```
COCO keypoint indexes:
        0: 'nose',
        1: 'left_eye',
        2: 'right_eye',
        3: 'left_ear',
        4: 'right_ear',
        5: 'left_shoulder',
        6: 'right_shoulder',
        7: 'left_elbow',
        8: 'right_elbow',
        9: 'left_wrist',
        10: 'right_wrist',
        11: 'left_hip',
        12: 'right_hip',
        13: 'left_knee',
        14: 'right_knee',
        15: 'left_ankle',
        16: 'right_ankle'
```
与Detection任务不同，KeyPoint任务的标注文件为`person_keypoints_train2017.json`和`person_keypoints_val2017.json`两个json文件。json文件中包含的`info`、`licenses`和`images`字段的含义与Detection相同，而`annotations`和`categories`则是不同的。
在`categories`字段中，除了给出类别，还给出了关键点的名称和互相之间的连接性。
在`annotations`字段中，标注了每一个实例的ID与所在图像，同时还有分割信息和关键点信息。其中与关键点信息较为相关的有：
- `keypoints`：`[x1,y1,v1 ...]`,是一个长度为17*3=51的List,每组表示了一个关键点的坐标与可见性，`v=0, x=0, y=0`表示该点不可见且未标注，`v=1`表示该点有标注但不可见，`v=2`表示该点有标注且可见。
- `bbox`: `[x1,y1,w,h]`表示该实例的检测框位置。
- `num_keypoints`: 表示该实例标注关键点的数目。


## MPII数据集
### MPII数据集的准备
请先通过[MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/#download)下载MPII数据集的图像与对应标注文件，并存放到`dataset/mpii`路径下。标注文件可以采用[mpii_annotations](https://download.openmmlab.com/mmpose/datasets/mpii_annotations.tar),已对应转换为json格式，完成后的目录结构为：
```
mpii
|── annotations
|   |── mpii_gt_val.mat
|   |── mpii_test.json
|   |── mpii_train.json
|   |── mpii_trainval.json
|   `── mpii_val.json
`── images
    |── 000001163.jpg
    |── 000003072.jpg
```
### MPII数据集的说明
在MPII中，关键点序号与部位的对应关系为：
```
MPII keypoint indexes:
        0: 'right_ankle',
        1: 'right_knee',
        2: 'right_hip',
        3: 'left_hip',
        4: 'left_knee',
        5: 'left_ankle',
        6: 'pelvis',
        7: 'thorax',
        8: 'upper_neck',
        9: 'head_top',
        10: 'right_wrist',
        11: 'right_elbow',
        12: 'right_shoulder',
        13: 'left_shoulder',
        14: 'left_elbow',
        15: 'left_wrist',
```
下面以一个解析后的标注信息为例，说明标注的内容，其中每条标注信息标注了一个人物实例：
```
{
    'joints_vis': [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    'gt_joints': [
        [-1.0, -1.0],
        [-1.0, -1.0],
        [-1.0, -1.0],
        [-1.0, -1.0],
        [-1.0, -1.0],
        [-1.0, -1.0],
        [-1.0, -1.0],
        [1232.0, 288.0],
        [1236.1271, 311.7755],
        [1181.8729, -0.77553],
        [692.0, 464.0],
        [902.0, 417.0],
        [1059.0, 247.0],
        [1405.0, 329.0],
        [1498.0, 613.0],
        [1303.0, 562.0]
    ],
    'image': '077096718.jpg',
    'scale': 9.516749,
    'center': [1257.0, 297.0]
}
```
- `joints_vis`：分别表示16个关键点是否标注，若为0，则对应序号的坐标也为`[-1.0, -1.0]`。
- `joints`：分别表示16个关键点的坐标。
- `image`：表示对应的图片文件。
- `center`：表示人物的大致坐标，用于定位人物在图像中的位置。
- `scale`：表示人物的比例，对应200px。


## 用户数据准备

### 数据格式转换

这里我们以`AIChallenger`数据集为例，展示如何将其他数据集对齐到COCO格式并加入关键点模型训练中。


`AI challenger`的标注格式如下：
```
AI Challenger Description:
        0: 'Right Shoulder',
        1: 'Right Elbow',
        2: 'Right Wrist',
        3: 'Left Shoulder',
        4: 'Left Elbow',
        5: 'Left Wrist',
        6: 'Right Hip',
        7: 'Right Knee',
        8: 'Right Ankle',
        9: 'Left Hip',
        10: 'Left Knee',
        11: 'Left Ankle',
        12: 'Head top',
        13: 'Neck'
```
1. 将`AI Challenger`点位序号，调整至与`COCO`数据集一致，（如`Right Shoulder`的序号由`0`调整到`13`。
2. 统一是否标注/可见的标志位信息，如`AI Challenger`中`标注且可见`需要由`1`调整到`2`。
3. 在该过程中，舍弃该数据集特有的点位（如`Neck`)；同时该数据集中没有的COCO点位（如`left_eye`等），对应设置为`v=0, x=0, y=0`，表示该未标注。
4. 为了避免不同数据集ID重复的问题，需要重新排列图像的`image_id`和`annotation id`。
5. 整理图像路径`file_name`，使其能够被正确访问到。

我们提供了整合`COCO`训练集和`AI Challenger`数据集的[标注文件](https://bj.bcebos.com/v1/paddledet/data/keypoint/aic_coco_train_cocoformat.json)，供您参考调整后的效果。

### 自定义数据训练

以[tinypose_256x192](../../../configs/keypoint/tiny_pose/README.md)为例来说明对于自定义数据如何修改：

#### 1、配置文件[tinypose_256x192.yml](../../../configs/keypoint/tiny_pose/tinypose_256x192.yml)

基本的修改内容及其含义如下：

```
num_joints: &num_joints 17    #自定义数据的关键点数量
train_height: &train_height 256   #训练图片尺寸-高度h
train_width: &train_width 192   #训练图片尺寸-宽度w
hmsize: &hmsize [48, 64]  #对应训练尺寸的输出尺寸，这里是输入[w,h]的1/4
flip_perm: &flip_perm [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]] #关键点定义中左右对称的关键点，用于flip增强。若没有对称结构在 TrainReader 的 RandomFlipHalfBodyTransform 一栏中 flip_pairs 后面加一行 "flip: False"（注意缩紧对齐）
num_joints_half_body: 8   #半身关键点数量，用于半身增强
prob_half_body: 0.3   #半身增强实现概率，若不需要则修改为0
upper_body_ids: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]    #上半身对应关键点id，用于半身增强中获取上半身对应的关键点。
```

上述是自定义数据时所需要的修改部分，完整的配置及含义说明可参考文件：[关键点配置文件说明](../KeyPointConfigGuide_cn.md)。

#### 2、其他代码修改（影响测试、可视化）
- keypoint_utils.py中的sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07,.87, .87, .89, .89]) / 10.0，表示每个关键点的确定范围方差，根据实际关键点可信区域设置，区域精确的一般0.25-0.5，例如眼睛。区域范围大的一般0.5-1.0，例如肩膀。若不确定建议0.75。
- visualizer.py中的draw_pose函数中的EDGES，表示可视化时关键点之间的连接线关系。
- pycocotools工具中的sigmas，同第一个keypoint_utils.py中的设置。用于coco指标评估时计算。

#### 3、数据准备注意
- 训练数据请按coco数据格式处理。需要包括关键点[Nx3]、检测框[N]标注。
- 请注意area>0，area=0时数据在训练时会被过滤掉。此外，由于COCO的评估机制，area较小的数据在评估时也会被过滤掉，我们建议在自定义数据时取`area = bbox_w * bbox_h`。
