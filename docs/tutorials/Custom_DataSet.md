# 如何训练自定义数据集

## 目录
- [1.数据准备](#1准备数据)
    - [将数据集转换为COCO格式](#方式一将数据集转换为COCO格式)
    - [将数据集转换为VOC格式](#方式二将数据集转换为VOC格式)
    - [添加新数据源](#方式三添加新数据源)
- [2.修改配置文件](#2修改配置文件)
- [3.(可选)生成Anchor](#3(可选)生成Anchor)

## 1.准备数据
如果数据符合COCO或VOC数据集格式，可以直接进入[2.修改配置文件](#2修改配置文件)，否则需要将数据集转换至COCO格式或VOC格式，或者添加新的数据源。

### 方式一：将数据集转换为COCO格式

在`tools/`中提供了`x2coco.py`用于将voc格式数据集、labelme标注的数据集或cityscape数据集转换为COCO数据集，例如:

（1）labelmes数据转换为COCO格式：
```bash
python tools/x2coco.py \
                --dataset_type labelme \
                --json_input_dir ./labelme_annos/ \
                --image_input_dir ./labelme_imgs/ \
                --output_dir ./cocome/ \
                --train_proportion 0.8 \
                --val_proportion 0.2 \
                --test_proportion 0.0
```

**参数说明：**

- `--dataset_type`：需要转换的数据格式，目前支持：’voc‘、’labelme‘和’cityscape‘
- `--json_input_dir`：使用labelme标注的json文件所在文件夹
- `--image_input_dir`：图像文件所在文件夹
- `--output_dir`：转换后的COCO格式数据集存放位置
- `--train_proportion`：标注数据中用于train的比例
- `--val_proportion`：标注数据中用于validation的比例
- `--test_proportion`：标注数据中用于infer的比例

（2）voc数据转换为COCO格式：
```bash
python tools/x2coco.py \
        --dataset_type voc \
        --voc_anno_dir path/to/VOCdevkit/VOC2007/Annotations/ \
        --voc_anno_list path/to/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt \
        --voc_label_list dataset/voc/label_list.txt \
        --voc_out_name voc_train.json
```

**参数说明：**

- `--dataset_type`：需要转换的数据格式，当前数据集是voc格式时，指定’voc‘即可。
- `--voc_anno_dir`：VOC数据转换为COCO数据集时的voc数据集标注文件路径。
例如：
```
├──Annotations/
   ├──    009881.xml
   ├──  009882.xml
   ├──  009886.xml
   ...
```
- `--voc_anno_list`：VOC数据转换为COCO数据集时的标注列表文件，文件中是文件名前缀列表，一般是`ImageSets/Main`下trainval.txt和test.txt文件。
例如：trainval.txt里的内容如下:
```
009881
009882
009886
...
```
- `--voc_label_list`：VOC数据转换为COCO数据集时的类别列表文件，文件中每一行表示一种物体类别。
例如：label_list.txt里的内容如下:
```
background
aeroplane
bicycle
...
```
- `--voc_out_name`：VOC数据转换为COCO数据集时的输出的COCO数据集格式json文件名。


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

如果数据集有新的格式需要添加进PaddleDetection中，您可自行参考数据处理文档中的[添加新数据源](../advanced_tutorials/READER.md#自定义数据集)文档部分，开发相应代码完成新的数据源支持，同时数据处理具体代码解析等可阅读[数据处理文档](../advanced_tutorials/READER.md)

## 2.修改配置文件
与数据集相关的配置文件存在于`configs/datasets`文件夹，目前PaddleDetection支持COCO和VOC格式等格式的数据集。如果你的数据集格式为coco格式，那么你需要修改`configs/datasets/coco_detection.yml`或者`configs/datasets/coco_instance.yml`。其中，coco_detection.yml与coco_instances.yml基本一致，唯一的区别在于coco_instances.yml包含实例分割的信息，而coco_detection.yml只包含框的信息。如果你的数据集为voc格式，那么你需要修改`configs/datasets/voc.yml`文件。下面coco_detection.yml文件为例，演示你在使用自定义数据集时需要修改的配置。
```
metric: COCO
num_classes: 80 # 修改num_classes为你的数据集的类别数，不包含背景类

TrainDataset:
  !COCODataSet
    image_dir: train2017 # 训练集的图片所在文件相对于dataset_dir的路径
    anno_path: annotations/instances_train2017.json # 训练集的标注文件相对于dataset_dir的路径
    dataset_dir: dataset/coco # 修改你的数据集所在路径，相对于PaddleDetection路径
    data_fields: ['image', 'gt_bbox', 'gt_class', 'is_crowd']

EvalDataset:
  !COCODataSet
    image_dir: val2017 # 验证集的图片所在文件夹相对于dataset_dir的路径
    anno_path: annotations/instances_val2017.json # 验证集的标注文件相对于dataset_dir的路径
    dataset_dir: dataset/coco # 数据集所在路径，相对于PaddleDetection路径

TestDataset:
  !ImageFolder
    anno_path: dataset/coco/annotations/instances_val2017.json # 验证集的标注文件所在路径，相对于PaddleDetection的路径
```
voc格式的数据集的修改与之类似，只需要修改voc.yml中的num_classes，dataset_dir, anno_path以及image_dir即可。

## 3.(可选)生成Anchor
在yolo系列模型中，可以运行`tools/anchor_cluster.py`来得到适用于你的数据集Anchor，使用方法如下：
``` bash
python tools/anchor_cluster.py -c configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml -n 9 -s 608 -m v2 -i 1000
```
目前`tools/anchor_cluster.py`支持的主要参数配置如下表所示：

|    参数    |    用途    |    默认值    |    备注    |
|:------:|:------:|:------:|:------:|
| -c/--config | 模型的配置文件 | 无默认值 | 必须指定 |
| -n/--n | 聚类的簇数 | 9 | Anchor的数目 |
| -s/--size | 图片的输入尺寸 | None | 若指定，则使用指定的尺寸，如果不指定, 则尝试从配置文件中读取图片尺寸 |
|  -m/--method  |  使用的Anchor聚类方法  |  v2  |  目前只支持yolov2/v5的聚类算法  |
|  -i/--iters  |  kmeans聚类算法的迭代次数  |  1000  | kmeans算法收敛或者达到迭代次数后终止 |
| -gi/--gen_iters |  遗传算法的迭代次数  | 1000 |  该参数只用于yolov5的Anchor聚类算法  |
| -t/--thresh|  Anchor尺度的阈值  | 0.25 | 该参数只用于yolov5的Anchor聚类算法 |

如仍有疑惑，欢迎给我们提issue。
