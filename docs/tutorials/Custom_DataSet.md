# 如何训练自定义数据集

## 目录
- [1.数据准备](#1.准备数据)
    - [将数据集转换为COCO格式](#方式一：将数据集转换为COCO格式)
    - [将数据集转换为VOC格式](#方式二：将数据集转换为VOC格式)
    - [添加新数据源](#方式三：添加新数据源)
- [2.选择模型](#2.选择模型)
- [3.修改参数配置](#3.修改参数配置)
- [4.开始训练](#4.开始训练)

## 1.准备数据
如果数据符合COCO或VOC数据集格式，可以直接进入[2.选择模型](#2.选择模型)，否则需要将数据集转换至COCO格式或VOC格式。

### 方式一：将数据集转换为COCO格式

在`./tools/`中提供了`x2coco.py`用于将labelme标注的数据集或cityscape数据集转换为COCO数据集:
```bash
python ./ppdet/data/tools/x2coco.py \
                --dataset_type labelme \
                --json_input_dir ./labelme_annos/ \
                --image_input_dir ./labelme_imgs/ \
                --output_dir ./cocome/ \
                --train_proportion 0.8 \
                --val_proportion 0.2 \
                --test_proportion 0.0 \
```
**参数说明：**

- `--dataset_type`：需要转换的数据格式，目前支持：’labelme‘和’cityscape‘
- `--json_input_dir`：使用labelme标注的json文件所在文件夹
- `--image_input_dir`：图像文件所在文件夹
- `--output_dir`：转换后的COCO格式数据集存放位置
- `--train_proportion`：标注数据中用于train的比例
- `--val_proportion`：标注数据中用于validation的比例
- `--test_proportion`：标注数据中用于infer的比例

### 方式二：将数据集转换为VOC格式

VOC数据集所必须的文件内容如下所示，数据集根目录需有`VOCdevkit/VOC2007`或`VOCdevkit/VOC2012`文件夹，该文件夹中需有`Annotations`,`JPEGImages`和`ImageSets/Main`三个子目录，`Annotations`存放图片标注的xml文件，`JPEGImages`存放数据集图片，`ImageSets/Main`存放训练trainval.txt和测试test.txt列表。
  ```
  VOCdevkit
  ├──VOC2007(或VOC2012)
  │   ├── Annotations
  │       ├── xxx.xml
  │   ├── JPEGImages
  │       ├── xxx.jpg
  │   ├── ImageSets
  │       ├── Main
  │           ├── trainval.txt
  │           ├── test.txt
  ```

执行以下脚本，将根据`ImageSets/Main`目录下的trainval.txt和test.txt文件在数据集根目录生成最终的`trainval.txt`和`test.txt`列表文件：
```shell
python dataset/voc/create_list.py -d path/to/dataset
```
**参数说明：**
- `-d`或`--dataset_dir`：VOC格式数据集所在文件夹路径


### 方式三：添加新数据源

如果数据集有新的格式需要添加进PaddleDetection中，您可自行参考数据处理文档中的[添加新数据源](../advanced_tutorials/READER.md#添加新数据源)文档部分，开发相应代码完成新的数据源支持，同时数据处理具体代码解析等可阅读[数据处理文档](../advanced_tutorials/READER.md)


## 2.选择模型

PaddleDetection中提供了丰富的模型库，具体可在[模型库](../MODEL_ZOO_cn.md)中查看各个模型的指标，您可依据实际部署算力的情况，选择合适的模型:

- 算力资源小时，推荐您使用[移动端模型](../featured_model/MOBILE_SIDE.md)，PaddleDetection中的移动端模型经过迭代优化，具有较高性价比。
- 算力资源强大时，推荐您使用[服务器端模型](../featured_model/SERVER_SIDE.md)，该模型是PaddleDetection提出的面向服务器端实用的目标检测方案。

同时也可以根据使用场景不同选择合适的模型：
- 当小物体检测时，推荐您使用两阶段检测模型，比如Faster RCNN系列模型，具体可在[模型库](../MODEL_ZOO_cn.md)中找到。
- 当在交通领域使用，如行人，车辆检测时，推荐您使用[特色垂类检测模型](../featured_model/CONTRIB_cn.md)。
- 当在竞赛中使用，推荐您使用竞赛冠军模型[CACascadeRCNN](../featured_model/champion_model/CACascadeRCNN.md)与[OIDV5_BASELINE_MODEL](../featured_model/champion_model/OIDV5_BASELINE_MODEL.md)。
- 当在人脸检测中使用，推荐您使用[人脸检测模型](../featured_model/FACE_DETECTION.md)。

同时也可以尝试PaddleDetection中开发的[YOLOv3增强模型](../featured_model/YOLOv3_ENHANCEMENT.md)、[YOLOv4模型](../featured_model/YOLO_V4.md)与[Anchor Free模型](../featured_model/ANCHOR_FREE_DETECTION.md)等。



## 3.修改参数配置

选择好模型后，需要在`configs`目录中找到对应的配置文件，为了适配在自定义数据集上训练，需要对参数配置做一些修改：

- 数据路径配置: 在yaml配置文件中，依据[1.数据准备](#1.准备数据)中准备好的路径，配置`TrainReader`、`EvalReader`和`TestReader`的路径。
    - COCO数据集：
     ```yaml
       dataset:
          !COCODataSet
          image_dir: val2017 # 图像数据基于数据集根目录的相对路径
          anno_path: annotations/instances_val2017.json  # 标注文件基于数据集根目录的相对路径
          dataset_dir: dataset/coco  # 数据集根目录
          with_background: true  # 背景是否作为一类标签，默认为true。
     ```
    - VOC数据集：
     ```yaml
       dataset:
          !VOCDataSet
          anno_path: trainval.txt   # 训练集列表文件基于数据集根目录的相对路径
          dataset_dir: dataset/voc  # 数据集根目录
          use_default_label: true   # 是否使用默认标签，默认为true。
          with_background: true  # 背景是否作为一类标签，默认为true。
     ```

**说明：** 如果您使用自己的数据集进行训练，需要将`use_default_label`设为`false`，并在数据集根目录中修改`label_list.txt`文件，添加自己的类别名，其中行号对应类别号。

- 类别数修改: 如果您自己的数据集类别数和COCO/VOC的类别数不同， 需修改yaml配置文件中类别数，`num_classes: XX`。
**注意：如果dataset中设置`with_background: true`，那么num_classes数必须是真实类别数+1（背景也算作1类）**

-  根据需要修改`LearningRate`相关参数:
    - 如果GPU卡数变化，依据lr，batch-size关系调整lr: [学习率调整策略](../FAQ.md#faq%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98)
    - 自己数据总数样本数和COCO不同，依据batch_size， 总共的样本数，换算总迭代次数`max_iters`，以及`LearningRate`中的`milestones`（学习率变化界限）。

- 预训练模型配置：通过在yaml配置文件中的`pretrain_weights: path/to/weights`参数可以配置路径，可以是链接或权重文件路径。可直接沿用配置文件中给出的在ImageNet数据集上的预训练模型。同时我们支持训练在COCO或Obj365数据集上的模型权重作为预训练模型，做迁移学习，详情可参考[迁移学习文档](../advanced_tutorials/TRANSFER_LEARNING_cn.md)。

## 4.开始训练

参数配置完成后，就可以开始训练模型了，具体可参考[训练/评估/预测](GETTING_STARTED_cn.md)入门文档。
如仍有疑惑，欢迎给我们提issue。
