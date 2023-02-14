简体中文 | [English](./keypoint_detection_en.md)

# 关键点检测任务二次开发

在实际场景中应用关键点检测算法，不可避免地会出现需要二次开发的需求。包括对目前的预训练模型效果不满意，希望优化模型效果；或是目前的关键点点位定义不能满足实际场景需求，希望新增或是替换关键点点位的定义，训练新的关键点模型。本文档将介绍如何在PaddleDetection中，对关键点检测算法进行二次开发。

## 数据准备

### 基本流程说明
在PaddleDetection中，目前支持的标注数据格式为`COCO`和`MPII`。这两个数据格式的详细说明，可以参考文档[关键点数据准备](../../tutorials/data/PrepareKeypointDataSet.md)。在这一步中，通过使用Labeme等标注工具，依照特征点序号标注对应坐标。并转化成对应可训练的标注格式。建议使用`COCO`格式进行。

### 合并数据集
为了扩展使用的训练数据，合并多个不同的数据集一起训练是一个很直观的解决手段，但不同的数据集往往对关键点的定义并不一致。合并数据集的第一步是需要统一不同数据集的点位定义，确定标杆点位，即最终模型学习的特征点类型，然后根据各个数据集的点位定义与标杆点位定义之间的关系进行调整。
- 在标杆点位中的点：调整点位序号，使其与标杆点位一致
- 未在标杆点位中的点：舍去
- 数据集缺少标杆点位中的点：对应将标注的标志位记为“未标注”

在[关键点数据准备](../../tutorials/data/PrepareKeypointDataSet.md)中，提供了如何合并`COCO`数据集和`AI Challenger`数据集，并统一为以`COCO`为标杆点位定义的案例说明，供参考。


## 模型优化

### 检测-跟踪模型优化
在PaddleDetection中，关键点检测能力支持Top-Down、Bottom-Up两套方案，Top-Down先检测主体，再检测局部关键点，优点是精度较高，缺点是速度会随着检测对象的个数增加，Bottom-Up先检测关键点再组合到对应的部位上，优点是速度快，与检测对象个数无关，缺点是精度较低。关于两种方案的详情及对应模型，可参考[关键点检测系列模型](../../../configs/keypoint/README.md)

当使用Top-Down方案时，模型效果依赖于前序的检测和跟踪效果，如果实际场景中不能准确检测到行人位置，会使关键点检测部分表现受限。如果在实际使用中遇到了上述问题，请参考[目标检测任务二次开发](./detection.md)以及[多目标跟踪任务二次开发](./pphuman_mot.md)对检测/跟踪模型进行优化。

### 使用符合场景的数据迭代
目前发布的关键点检测算法模型主要在`COCO`/ `AI Challenger`等开源数据集上迭代，这部分数据集中可能缺少与实际任务较为相似的监控场景（视角、光照等因素）、体育场景（存在较多非常规的姿态）。使用更符合实际任务场景的数据进行训练，有助于提升模型效果。

### 使用预训练模型迭代
关键点模型的数据的标注复杂度较大，直接使用模型从零开始在业务数据集上训练，效果往往难以满足需求。在实际工程中使用时，建议加载已经训练好的权重，通常能够对模型精度有较大提升，以`HRNet`为例，使用方法如下：
```bash
python tools/train.py -c configs/keypoint/hrnet/hrnet_w32_256x192.yml -o pretrain_weights=https://paddledet.bj.bcebos.com/models/keypoint/hrnet_w32_256x192.pdparams
```
在加载预训练模型后，可以适当减小初始学习率和最终迭代轮数, 建议初始学习率取默认配置值的1/2至1/5，并可开启`--eval`观察迭代过程中AP值的变化。


### 遮挡数据增强
关键点任务中有较多遮挡问题，包括自身遮挡与不同目标之间的遮挡。

1. 检测模型优化（仅针对Top-Down方案）

参考[目标检测任务二次开发](./detection.md)，提升检测模型在复杂场景下的效果。

2. 关键点数据增强

在关键点模型训练中增加遮挡的数据增强，参考[PP-TinyPose](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/configs/keypoint/tiny_pose/tinypose_256x192.yml#L100)。有助于模型提升这类场景下的表现。

### 对视频预测进行平滑处理
关键点模型是在图片级别的基础上进行训练和预测的，对于视频类型的输入也是将视频拆分为帧进行预测。帧与帧之间虽然内容大多相似，但微小的差异仍然可能导致模型的输出发生较大的变化，表现为虽然预测的坐标大体正确，但视觉效果上有较大的抖动问题。通过添加滤波平滑处理，将每一帧预测的结果与历史结果综合考虑，得到最终的输出结果，可以有效提升视频上的表现。该部分内容可参考[滤波平滑处理](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.6/deploy/python/det_keypoint_unite_infer.py#L206)。


## 新增或修改关键点点位定义

### 数据准备
根据前述说明，完成数据的准备，放置于`{root of PaddleDetection}/dataset`下。

<details>
<summary><b> 标注文件示例</b></summary>

一个标注文件示例如下：

```
self_dataset/
├── train_coco_joint.json  # 训练集标注文件
├── val_coco_joint.json    # 验证集标注文件
├── images/                # 存放图片文件
    ├── 0.jpg
    ├── 1.jpg
    ├── 2.jpg  
```
其中标注文件中需要注意的改动如下：
```json
{
    "images": [
        {
            "file_name": "images/0.jpg",
            "id": 0,       # 图片id，注意不可重复
            "height": 1080,
            "width": 1920
        },
        {
            "file_name": "images/1.jpg",
            "id": 1,
            "height": 1080,
            "width": 1920
        },
        {
            "file_name": "images/2.jpg",
            "id": 2,
            "height": 1080,
            "width": 1920
        },
    ...

    "categories": [
        {
            "supercategory": "person",
            "id": 1,
            "name": "person",
            "keypoints": [   # 点位序号的名称
                "point1",
                "point2",
                "point3",
                "point4",
                "point5",
            ],
            "skeleton": [    # 点位构成的骨骼, 训练中非必要
                [
                    1,
                    2
                ],
                [
                    1,
                    3
                ],
                [
                    2,
                    4
                ],
                [
                    3,
                    5
                ]
            ]
    ...

    "annotations": [
        {
            {
            "category_id": 1, # 实例所属类别
            "num_keypoints": 3, # 该实例已标注点数量
            "bbox": [         # 检测框位置,格式为x, y, w, h
                799,
                575,
                55,
                185
            ],
            # N*3 的列表，内容为x, y, v。
            "keypoints": [  
                807.5899658203125,
                597.5455322265625,
                2,
                0,  
                0,
                0,            # 未标注的点记为0，0，0
                805.8563232421875,
                592.3446655273438,
                2,
                816.258056640625,
                594.0783081054688,
                2,
                0,
                0,
                0
            ]
            "id": 1,      # 实例id，不可重复
            "image_id": 8,  # 实例所在图像的id，可重复。此时代表一张图像上存在多个目标
            "iscrowd": 0,   # 是否遮挡，为0时参与训练
            "area": 10175   # 实例所占面积，可简单取为w * h。注意为0时会跳过，过小时在eval时会被忽略

    ...
```

</details>


### 配置文件设置

在配置文件中，完整的含义参考[config yaml配置项说明](../../tutorials/KeyPointConfigGuide_cn.md)。以[HRNet模型配置](../../../configs/keypoint/hrnet/hrnet_w32_256x192.yml)为例，重点需要关注的内容如下：

<details>
<summary><b> 配置文件示例</b></summary>

一个配置文件的示例如下

```yaml
use_gpu: true
log_iter: 5
save_dir: output
snapshot_epoch: 10
weights: output/hrnet_w32_256x192/model_final
epoch: 210
num_joints: &num_joints 5 # 预测的点数与定义点数量一致
pixel_std: &pixel_std 200
metric: KeyPointTopDownCOCOEval
num_classes: 1  
train_height: &train_height 256
train_width: &train_width 192
trainsize: &trainsize [*train_width, *train_height]
hmsize: &hmsize [48, 64]
flip_perm: &flip_perm [[1, 2], [3, 4]]  # 注意只有含义上镜像对称的点才写到这里

...

# 保证dataset_dir + anno_path 能正确定位到标注文件位置
# 保证dataset_dir + image_dir + 标注文件中的图片路径能正确定位到图片
TrainDataset:
  !KeypointTopDownCocoDataset
    image_dir: images
    anno_path: train_coco_joint.json
    dataset_dir: dataset/self_dataset
    num_joints: *num_joints
    trainsize: *trainsize
    pixel_std: *pixel_std
    use_gt_bbox: True


EvalDataset:
  !KeypointTopDownCocoDataset
    image_dir: images
    anno_path: val_coco_joint.json
    dataset_dir: dataset/self_dataset
    bbox_file: bbox.json
    num_joints: *num_joints
    trainsize: *trainsize
    pixel_std: *pixel_std
    use_gt_bbox: True
    image_thre: 0.0
```
</details>

### 模型训练及评估
#### 模型训练
通过如下命令启动训练：
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m paddle.distributed.launch tools/train.py -c configs/keypoint/hrnet/hrnet_w32_256x192.yml
```

#### 模型评估
训练好模型之后，可以通过以下命令实现对模型指标的评估:
```bash
python3 tools/eval.py -c configs/keypoint/hrnet/hrnet_w32_256x192.yml
```

注意：由于测试依赖pycocotools工具，其默认为`COCO`数据集的17点，如果修改后的模型并非预测17点，直接使用评估命令会报错。
需要修改以下内容以获得正确的评估结果：
- [sigma列表](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.6/ppdet/modeling/keypoint_utils.py#L219)，表示每个关键点的范围方差，越大则容忍度越高。其长度与预测点数一致。根据实际关键点可信区域设置，区域精确的一般0.25-0.5，例如眼睛。区域范围大的一般0.5-1.0，例如肩膀。若不确定建议0.75。
- [pycocotools sigma列表](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py#L523)，含义及内容同上，取值与sigma列表一致。

### 模型导出及预测
#### Top-Down模型联合部署
```shell
#导出关键点模型
python tools/export_model.py -c configs/keypoint/hrnet/hrnet_w32_256x192.yml -o weights={path_to_your_weights}

#detector 检测 + keypoint top-down模型联合部署（联合推理只支持top-down方式）
python deploy/python/det_keypoint_unite_infer.py --det_model_dir=output_inference/ppyolo_r50vd_dcn_2x_coco/ --keypoint_model_dir=output_inference/hrnet_w32_256x192/ --video_file=../video/xxx.mp4  --device=gpu
```
- 注意目前PP-Human中使用的为该方案

#### Bottom-Up模型独立部署
```shell
#导出模型
python tools/export_model.py -c configs/keypoint/higherhrnet/higherhrnet_hrnet_w32_512.yml -o weights=output/higherhrnet_hrnet_w32_512/model_final.pdparams

#部署推理
python deploy/python/keypoint_infer.py --model_dir=output_inference/higherhrnet_hrnet_w32_512/ --image_file=./demo/000000014439_640x640.jpg --device=gpu --threshold=0.5

```
