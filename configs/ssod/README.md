简体中文 | [English](README_en.md)

# Semi-Supervised Object Detection (SSOD) 半监督目标检测

## 内容
- [简介](#简介)
- [模型库](#模型库)
- [数据集准备](#数据集准备)
- [引用](#引用)

## 简介
半监督目标检测(SSOD)是**同时使用有标注数据和无标注数据**进行训练的目标检测，既可以极大地节省标注成本，也可以充分利用无标注数据进一步提高检测精度。


## 模型库

### [Baseline](baseline)

**纯监督数据**模型的训练和模型库，请参照[Baseline](baseline)；



## 数据集准备

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


<details>
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
