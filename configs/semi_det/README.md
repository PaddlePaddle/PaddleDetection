简体中文 | [English](README_en.md)

# Semi-Supervised Detection (Semi DET) 半监督检测

## 内容
- [简介](#简介)
- [模型库](#模型库)
    - [Baseline](#Baseline)
    - [DenseTeacher](#DenseTeacher)
- [半监督数据集准备](#半监督数据集准备)
- [半监督检测配置](#半监督检测配置)
    - [训练集配置](#训练集配置)
    - [预训练配置](#预训练配置)
    - [全局配置](#全局配置)
    - [模型配置](#模型配置)
    - [数据增强配置](#数据增强配置)
    - [其他配置](#其他配置)
- [使用说明](#使用说明)
    - [训练](#训练)
    - [评估](#评估)
    - [预测](#预测)
    - [部署](#部署)
- [引用](#引用)

## 简介
半监督目标检测(Semi DET)是**同时使用有标注数据和无标注数据**进行训练的目标检测，既可以极大地节省标注成本，也可以充分利用无标注数据进一步提高检测精度。PaddleDetection团队复现了[DenseTeacher](denseteacher)半监督检测算法，用户可以下载使用。

## 模型库

### [Baseline](baseline)

**纯监督数据**模型的训练和模型库，请参照[Baseline](baseline)；


### [DenseTeacher](denseteacher)

|      模型       |  监督数据比例 |        Sup Baseline     |    Sup Epochs (Iters)   |  Sup mAP<sup>val<br>0.5:0.95 | Semi mAP<sup>val<br>0.5:0.95 |  Semi Epochs (Iters)  |  模型下载  |   配置文件   |
| :------------: | :---------: | :---------------------: | :---------------------: |:---------------------------: |:----------------------------: | :------------------: |:--------: |:----------: |
| DenseTeacher-FCOS     |   5% |   [sup_config](./baseline/fcos_r50_fpn_2x_coco_sup005.yml)    |  24 (8712)  | 21.3 |  **30.6**  | 240 (87120)   | [download](https://paddledet.bj.bcebos.com/models/denseteacher_fcos_r50_fpn_coco_semi005.pdparams) | [config](denseteacher/denseteacher_fcos_r50_fpn_coco_semi005.yml) |
| DenseTeacher-FCOS     |   10% |   [sup_config](./baseline/fcos_r50_fpn_2x_coco_sup010.yml)    | 24 (17424)  | 26.3 |  **35.1**  | 240 (174240)  | [download](https://paddledet.bj.bcebos.com/models/denseteacher_fcos_r50_fpn_coco_semi010.pdparams) | [config](denseteacher/denseteacher_fcos_r50_fpn_coco_semi010.yml) |
| DenseTeacher-FCOS(LSJ)|   10% |   [sup_config](./baseline/fcos_r50_fpn_2x_coco_sup010.yml)    | 24 (17424)  | 26.3 |  **37.1(LSJ)**  | 240 (174240)  | [download](https://paddledet.bj.bcebos.com/models/denseteacher_fcos_r50_fpn_coco_semi010_lsj.pdparams) | [config](denseteacher/denseteacher_fcos_r50_fpn_coco_semi010_lsj.yml) |
| DenseTeacher-FCOS |100%(full)|   [sup_config](./../fcos/fcos_r50_fpn_iou_multiscale_2x_coco.ymll) | 24 (175896) | 42.6 | **44.2** | 24 (175896)| [download](https://paddledet.bj.bcebos.com/models/denseteacher_fcos_r50_fpn_coco_full.pdparams) | [config](denseteacher/denseteacher_fcos_r50_fpn_coco_full.yml) |


## 半监督数据集准备

半监督目标检测**同时需要有标注数据和无标注数据**，且无标注数据量一般**远多于有标注数据量**。
对于COCO数据集一般有两种常规设置：

（1）抽取部分比例的原始训练集`train2017`作为标注数据和无标注数据；

从`train2017`中按固定百分比（1%、2%、5%、10%等）抽取，由于抽取方法会对半监督训练的结果影响较大，所以采用五折交叉验证来评估。运行数据集划分制作的脚本如下：
```bash
python tools/gen_semi_coco.py
```
会按照 1%、2%、5%、10% 的监督数据比例来划分`train2017`全集，为了交叉验证每一种划分会随机重复5次，生成的半监督标注文件如下：
- 标注数据集标注：`instances_train2017.{fold}@{percent}.json`
- 无标注数据集标注：`instances_train2017.{fold}@{percent}-unlabeled.json`
其中，`fold` 表示交叉验证，`percent` 表示有标注数据的百分比。

注意如果根据`txt_file`生成，需要下载`COCO_supervision.txt`:
```shell
wget https://bj.bcebos.com/v1/paddledet/data/coco/COCO_supervision.txt
```

（2）使用全量原始训练集`train2017`作为有标注数据 和 全量原始无标签图片集`unlabeled2017`作为无标注数据；


### 下载链接

PaddleDetection团队提供了COCO数据集全部的标注文件，请下载并解压存放至对应目录:

```shell
# 下载COCO全量数据集图片和标注
# 包括 train2017, val2017, annotations
wget https://bj.bcebos.com/v1/paddledet/data/coco.tar

# 下载PaddleDetection团队整理的COCO部分比例数据的标注文件
wget https://bj.bcebos.com/v1/paddledet/data/coco/semi_annotations.zip

# unlabeled2017是可选，如果不需要训‘full’则无需下载
# 下载COCO全量 unlabeled 无标注数据集
wget https://bj.bcebos.com/v1/paddledet/data/coco/unlabeled2017.zip
wget https://bj.bcebos.com/v1/paddledet/data/coco/image_info_unlabeled2017.zip
# 下载转换完的 unlabeled2017 无标注json文件
wget https://bj.bcebos.com/v1/paddledet/data/coco/instances_unlabeled2017.zip
```

如果需要用到COCO全量unlabeled无标注数据集，需要将原版的`image_info_unlabeled2017.json`进行格式转换，运行以下代码:

<details>
<summary> COCO unlabeled 标注转换代码：</summary>

```python
import json
anns_train = json.load(open('annotations/instances_train2017.json', 'r'))
anns_unlabeled = json.load(open('annotations/image_info_unlabeled2017.json', 'r'))
unlabeled_json = {
  'images': anns_unlabeled['images'],
  'annotations': [],
  'categories': anns_train['categories'],
}
path = 'annotations/instances_unlabeled2017.json'
with open(path, 'w') as f:
  json.dump(unlabeled_json, f)
```

</details>


<details open>
<summary> 解压后的数据集目录如下：</summary>

```
PaddleDetection
├── dataset
│   ├── coco
│   │   ├── annotations
│   │   │   ├── instances_train2017.json
│   │   │   ├── instances_unlabeled2017.json
│   │   │   ├── instances_val2017.json
│   │   ├── semi_annotations
│   │   │   ├── instances_train2017.1@1.json
│   │   │   ├── instances_train2017.1@1-unlabeled.json
│   │   │   ├── instances_train2017.1@2.json
│   │   │   ├── instances_train2017.1@2-unlabeled.json
│   │   │   ├── instances_train2017.1@5.json
│   │   │   ├── instances_train2017.1@5-unlabeled.json
│   │   │   ├── instances_train2017.1@10.json
│   │   │   ├── instances_train2017.1@10-unlabeled.json
│   │   ├── train2017
│   │   ├── unlabeled2017
│   │   ├── val2017
```

</details>

## 半监督检测配置

配置半监督检测，需要基于选用的**基础检测器**的配置文件，如：

```python
_BASE_: [
  '../../fcos/fcos_r50_fpn_iou_multiscale_2x_coco.yml',
  '../_base_/coco_detection_percent_10.yml',
]
log_iter: 50
snapshot_epoch: 5
epochs: &epochs 240
weights: output/denseteacher_fcos_r50_fpn_coco_semi010/model_final
```
并依次做出如下几点改动：

### 训练集配置

首先可以直接引用已经配置好的半监督训练集，如：

```python
_BASE_: [
  '../_base_/coco_detection_percent_10.yml',
]
```

具体来看，构建半监督数据集，需要同时配置监督数据集`TrainDataset`和无监督数据集`UnsupTrainDataset`的路径，**注意必须选用`SemiCOCODataSet`类而不是`COCODataSet`类**，如以下所示:

**COCO-train2017部分比例数据集**：

```python
# partial labeled COCO, use `SemiCOCODataSet` rather than `COCODataSet`
TrainDataset:
  !SemiCOCODataSet
    image_dir: train2017
    anno_path: semi_annotations/instances_train2017.1@10.json
    dataset_dir: dataset/coco
    data_fields: ['image', 'gt_bbox', 'gt_class', 'is_crowd']

# partial unlabeled COCO, use `SemiCOCODataSet` rather than `COCODataSet`
UnsupTrainDataset:
  !SemiCOCODataSet
    image_dir: train2017
    anno_path: semi_annotations/instances_train2017.1@10-unlabeled.json
    dataset_dir: dataset/coco
    data_fields: ['image']
    supervised: False
```

或者 **COCO-train2017 full全量数据集**：

```python
# full labeled COCO, use `SemiCOCODataSet` rather than `COCODataSet`
TrainDataset:
  !SemiCOCODataSet
    image_dir: train2017
    anno_path: annotations/instances_train2017.json
    dataset_dir: dataset/coco
    data_fields: ['image', 'gt_bbox', 'gt_class', 'is_crowd']

# full unlabeled COCO, use `SemiCOCODataSet` rather than `COCODataSet`
UnsupTrainDataset:
  !SemiCOCODataSet
    image_dir: unlabeled2017
    anno_path: annotations/instances_unlabeled2017.json
    dataset_dir: dataset/coco
    data_fields: ['image']
    supervised: False
```

验证集`EvalDataset`和测试集`TestDataset`的配置**不需要更改**，且还是采用`COCODataSet`类。


### 预训练配置

```python
### pretrain and warmup config, choose one and comment another
pretrain_weights: https://paddledet.bj.bcebos.com/models/pretrained/ResNet50_cos_pretrained.pdparams
semi_start_iters: 5000
ema_start_iters: 3000
use_warmup: &use_warmup True
```

**注意:**
 - `Dense Teacher`原文使用`R50-va-caffe`预训练，PaddleDetection中默认使用`R50-vb`预训练，如果使用`R50-vd`结合[SSLD](../../../docs/feature_models/SSLD_PRETRAINED_MODEL.md)的预训练模型，可进一步显著提升检测精度，同时backbone部分配置也需要做出相应更改，如：
 ```python
  pretrain_weights:  https://paddledet.bj.bcebos.com/models/pretrained/ResNet50_vd_ssld_v2_pretrained.pdparams
  ResNet:
    depth: 50
    variant: d
    norm_type: bn
    freeze_at: 0
    return_idx: [1, 2, 3]
    num_stages: 4
    lr_mult_list: [0.05, 0.05, 0.1, 0.15]
```

### 全局配置

需要在配置文件中添加如下全局配置，并且注意 DenseTeacher 模型需要使用`use_simple_ema: True`而不是`use_ema: True`：

```python
### global config
use_simple_ema: True
ema_decay: 0.9996
ssod_method: DenseTeacher
DenseTeacher:
  train_cfg:
    sup_weight: 1.0
    unsup_weight: 1.0
    loss_weight: {distill_loss_cls: 4.0, distill_loss_box: 1.0, distill_loss_quality: 1.0}
    concat_sup_data: True
    suppress: linear
    ratio: 0.01
    gamma: 2.0
  test_cfg:
    inference_on: teacher
```

### 模型配置

如果没有特殊改动，则直接继承自基础检测器里的模型配置。
以 `DenseTeacher` 为例，选择 `fcos_r50_fpn_iou_multiscale_2x_coco.yml` 作为**基础检测器**进行半监督训练，**teacher网络的结构和student网络的结构均为基础检测器的结构，且结构相同**。

```python
_BASE_: [
  '../../fcos/fcos_r50_fpn_iou_multiscale_2x_coco.yml',
]
```

### 数据增强配置

构建半监督训练集的Reader，需要在原先`TrainReader`的基础上，新增加`weak_aug`,`strong_aug`,`sup_batch_transforms`和`unsup_batch_transforms`，并且需要注意：
- 如果有`NormalizeImage`，需要单独从`sample_transforms`中抽出来放在`weak_aug`和`strong_aug`中；
- `sample_transforms`为**公用的基础数据增强**；
- 完整的弱数据增强为`sample_transforms + weak_aug`，完整的强数据增强为`sample_transforms + strong_aug`；

如以下所示:

原纯监督模型的`TrainReader`：
```python
TrainReader:
  sample_transforms:
    - Decode: {}
    - RandomResize: {target_size: [[640, 1333], [672, 1333], [704, 1333], [736, 1333], [768, 1333], [800, 1333]], keep_ratio: True, interp: 1}
    - RandomFlip: {}
    - NormalizeImage: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225], is_scale: True}
  batch_transforms:
    - Permute: {}
    - PadBatch: {pad_to_stride: 32}
    - Gt2FCOSTarget:
        object_sizes_boundary: [64, 128, 256, 512]
        center_sampling_radius: 1.5
        downsample_ratios: [8, 16, 32, 64, 128]
        norm_reg_targets: True
  batch_size: 2
  shuffle: True
  drop_last: True
```

更改后的半监督TrainReader：

```python
### reader config
SemiTrainReader:
  sample_transforms:
    - Decode: {}
    - RandomResize: {target_size: [[640, 1333], [672, 1333], [704, 1333], [736, 1333], [768, 1333], [800, 1333]], keep_ratio: True, interp: 1}
    - RandomFlip: {}
  weak_aug:
    - NormalizeImage: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225], is_scale: true}
  strong_aug:
    - StrongAugImage: {transforms: [
        RandomColorJitter: {prob: 0.8, brightness: 0.4, contrast: 0.4, saturation: 0.4, hue: 0.1},
        RandomErasingCrop: {},
        RandomGaussianBlur: {prob: 0.5, sigma: [0.1, 2.0]},
        RandomGrayscale: {prob: 0.2},
      ]}
    - NormalizeImage: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225], is_scale: true}
  sup_batch_transforms:
    - Permute: {}
    - PadBatch: {pad_to_stride: 32}
    - Gt2FCOSTarget:
        object_sizes_boundary: [64, 128, 256, 512]
        center_sampling_radius: 1.5
        downsample_ratios: [8, 16, 32, 64, 128]
        norm_reg_targets: True
  unsup_batch_transforms:
    - Permute: {}
    - PadBatch: {pad_to_stride: 32}
  sup_batch_size: 2
  unsup_batch_size: 2
  shuffle: True
  drop_last: True
```

### 其他配置

训练epoch数需要和全量数据训练时换算总iter数保持一致，如全量训练24 epoch(换算约为180k个iter)，则10%监督数据的半监督训练，总epoch数需要为240 epoch左右(换算约为180k个iter)。示例如下：

```python
### other config
epoch: 240
LearningRate:
  base_lr: 0.01
  schedulers:
  - !PiecewiseDecay
    gamma: 0.1
    milestones: 240
    use_warmup: True
  - !LinearWarmup
    start_factor: 0.001
    steps: 1000

OptimizerBuilder:
  optimizer:
    momentum: 0.9
    type: Momentum
  regularizer:
    factor: 0.0001
    type: L2
  clip_grad_by_value: 1.0
```


## 使用说明

仅训练时必须使用半监督检测的配置文件去训练，评估、预测、部署也可以按基础检测器的配置文件去执行。

### 训练

```bash
# 单卡训练 (不推荐，需按线性比例相应地调整学习率)
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/semi_det/denseteacher/denseteacher_fcos_r50_fpn_coco_semi010.yml --eval

# 多卡训练
python -m paddle.distributed.launch --log_dir=denseteacher_fcos_semi010/ --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/semi_det/denseteacher/denseteacher_fcos_r50_fpn_coco_semi010.yml --eval
```

### 评估

```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/semi_det/denseteacher/denseteacher_fcos_r50_fpn_coco_semi010.yml -o weights=output/denseteacher_fcos_r50_fpn_coco_semi010/model_final.pdparams
```

### 预测

```bash
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/semi_det/denseteacher/denseteacher_fcos_r50_fpn_coco_semi010.yml -o weights=output/denseteacher_fcos_r50_fpn_coco_semi010/model_final.pdparams --infer_img=demo/000000014439.jpg
```

### 部署

部署可以使用半监督检测配置文件，也可以使用基础检测器的配置文件去部署和使用。

```bash
# 导出模型
CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c configs/semi_det/denseteacher/denseteacher_fcos_r50_fpn_coco_semi010.yml -o weights=https://paddledet.bj.bcebos.com/models/denseteacher_fcos_r50_fpn_coco_semi010.pdparams

# 导出权重预测
CUDA_VISIBLE_DEVICES=0 python deploy/python/infer.py --model_dir=output_inference/denseteacher_fcos_r50_fpn_coco_semi010 --image_file=demo/000000014439_640x640.jpg --device=GPU

# 部署测速
CUDA_VISIBLE_DEVICES=0 python deploy/python/infer.py --model_dir=output_inference/denseteacher_fcos_r50_fpn_coco_semi010 --image_file=demo/000000014439_640x640.jpg --device=GPU --run_benchmark=True # --run_mode=trt_fp16

# 导出ONNX
paddle2onnx --model_dir output_inference/denseteacher_fcos_r50_fpn_coco_semi010/ --model_filename model.pdmodel --params_filename model.pdiparams --opset_version 12 --save_file denseteacher_fcos_r50_fpn_coco_semi010.onnx
```


## 引用

```
 @article{denseteacher2022,
  title={Dense Teacher: Dense Pseudo-Labels for Semi-supervised Object Detection},
  author={Hongyu Zhou, Zheng Ge, Songtao Liu, Weixin Mao, Zeming Li, Haiyan Yu, Jian Sun},
  journal={arXiv preprint arXiv:2207.02541},
  year={2022}
}
```
