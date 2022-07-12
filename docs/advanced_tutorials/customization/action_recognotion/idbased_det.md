# 基于人体id的检测开发

## 环境准备

基于人体id的检测方案是直接使用[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)的功能进行模型训练的。请按照[安装说明](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/docs/tutorials/INSTALL_cn.md)完成环境安装，以进行后续的模型训练及使用流程。

## 数据准备
基于检测的行为识别方案中，数据准备的流程与一般的检测模型一致，详情可参考[目标检测数据准备](../../tutorials/data/PrepareDetDataSet.md)。将图像和标注数据组织成PaddleDetection中支持的格式之一即可。

**注意** ： 在实际使用的预测过程中，使用的是单人图像进行预测，因此在训练过程中建议将图像裁剪为单人图像，再进行烟头检测框的标注，以提升准确率。


## 模型优化

### 检测-跟踪模型优化
基于检测的行为识别模型效果依赖于前序的检测和跟踪效果，如果实际场景中不能准确检测到行人位置，或是难以正确在不同帧之间正确分配人物ID，都会使行为识别部分表现受限。如果在实际使用中遇到了上述问题，请参考[目标检测任务二次开发](../detection.md)以及[多目标跟踪任务二次开发](../mot.md)对检测/跟踪模型进行优化。


### 更大的分辨率
烟头的检测在监控视角下是一个典型的小目标检测问题，使用更大的分辨率有助于提升模型整体的识别率

### 预训练模型
加入小目标场景数据集VisDrone下的预训练模型进行训练，模型mAP由38.1提升到39.7。

## 新增行为
### 数据准备
参考[目标检测数据准备](../../tutorials/data/PrepareDetDataSet.md)完成训练数据准备。

准备完成后，数据路径为
```
dataset/smoking
├── smoking # 存放所有的图片
│   ├── 1.jpg
│   ├── 2.jpg
├── smoking_test_cocoformat.json # 测试标注文件
├── smoking_train_cocoformat.json # 训练标注文件
```

以`COCO`格式为例，完成后的json标注文件内容如下：

```
# images字段下包含了图像的路径，id及对应宽高信息
  "images": [
    {
      "file_name": "smoking/1.jpg",
      "id": 0,    # 此处id为图片id序号，不要重复
      "height": 437,
      "width": 212
    },
    {
      "file_name": "smoking/2.jpg",
      "id": 1,
      "height": 655,
      "width": 365
    },

 ...

# categories 字段下包含所有类别信息，如果希望新增更多的检测类别，请在这里增加, 示例如下。
  "categories": [
    {
      "supercategory": "cigarette",
      "id": 1,
      "name": "cigarette"
    },
    {
      "supercategory": "Class_Defined_by_Yourself",
      "id": 2,
      "name": "Class_Defined_by_Yourself"
    },

  ...

# annotations 字段下包含了所有目标实例的信息，包括类别，检测框坐标, id, 所属图像id等信息
  "annotations": [
    {
      "category_id": 1,  # 对应定义的类别，在这里1代表cigarette
      "bbox": [
        97.0181345931,
        332.7033243081,
        7.5943999555,
        16.4545332369
      ],
      "id": 0,           # 此处id为实例的id序号，不要重复
      "image_id": 0,     # 此处为实例所在图片的id序号，可能重复，此时即一张图片上有多个实例对象
      "iscrowd": 0,
      "area": 124.96230648208665
    },
    {
      "category_id": 2, # 对应定义的类别，在这里2代表Class_Defined_by_Yourself
      "bbox": [
        114.3895698372,
        221.9131122343,
        25.9530363697,
        50.5401234568
      ],
      "id": 1,
      "image_id": 1,
      "iscrowd": 0,
      "area": 1311.6696622034585
```

### 配置文件设置
参考[配置文件](../../../../configs/pphuman/ppyoloe_crn_s_80e_smoking_visdrone.yml), 其中需要关注重点如下：

```
metric: COCO
num_classes: 1 # 如果新增了更多的类别，请对应修改此处

# 正确设置image_dir，anno_path，dataset_dir
# 保证dataset_dir + anno_path 能正确对应标注文件的路径
# 保证dataset_dir + image_dir + 标注文件中的图片路径可以正确对应到图片路径
TrainDataset:
  !COCODataSet
    image_dir: ""
    anno_path: smoking_train_cocoformat.json
    dataset_dir: dataset/smoking
    data_fields: ['image', 'gt_bbox', 'gt_class', 'is_crowd']

EvalDataset:
  !COCODataSet
    image_dir: ""
    anno_path: smoking_test_cocoformat.json
    dataset_dir: dataset/smoking

TestDataset:
  !ImageFolder
    anno_path: smoking_test_cocoformat.json
    dataset_dir: dataset/smoking
```

### 模型训练及测试
- 模型训练: 参考[PP-YOLOE](../../../configs/ppyoloe/README_cn.md)，执行下列步骤实现

```bash
# At Root of PaddleDetection

python -m paddle.distributed.launch --gpus 0,1,2,3  tools/train.py -c configs/pphuman/ppyoloe_crn_s_80e_smoking_visdrone.yml --eval
```

### 模型导出
注意：如果在Tensor-RT环境下预测, 请开启`-o trt=True`以获得更好的性能
```bash
# At Root of PaddleDetection

python tools/export_model.py -c configs/pphuman/ppyoloe_crn_s_80e_smoking_visdrone.yml -o weights=output/ppyoloe_crn_s_80e_smoking_visdrone/best_model trt=True
```

导出模型后，可以得到：
```
ppyoloe_crn_s_80e_smoking_visdrone/
├── infer_cfg.yml
├── model.pdiparams
├── model.pdiparams.info
└── model.pdmodel
```

至此，即可使用PP-Human进行实际预测了。
