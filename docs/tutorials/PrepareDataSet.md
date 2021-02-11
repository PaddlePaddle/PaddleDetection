# 如何准备训练数据
## 目录
- [目标检测数据说明](#目标检测数据说明)
- [准备训练数据](#准备训练数据)
    - [VOC数据数据](#VOC数据数据)
        - [VOC数据集下载](#VOC数据集下载)
        - [VOC数据标注文件介绍](#VOC数据标注文件介绍)
    - [COCO数据数据](#COCO数据数据)
        - [COCO数据集下载](#COCO数据下载)
        - [COCO数据标注文件介绍](#COCO数据标注文件介绍)
    - [用户数据](#用户数据)
        - [用户数据转成VOC数据](#用户数据转成VOC数据)
        - [用户数据转成COCO数据](#用户数据转成COCO数据)
        - [用户数据自定义reader](#用户数据自定义reader)
    - [用户数据数据转换示例](#用户数据数据转换示例)

### 目标检测数据说明  
目标检测的数据比分类复杂，一张图像中，需要标记出各个目标区域的位置和类别。

一般的目标区域位置用一个矩形框来表示，一般用以下3种方式表达：

|         表达方式    |                 说明               |
| :----------------: | :--------------------------------: |
|     x1,y1,x2,y2    | (x1,y1)为左上角坐标，(x2,y2)为右下角坐标  |  
|       x,y,w,h      | (x,y)为左上角坐标，w为目标区域宽度，h为目标区域高度  |
|     xc,yc,w,h    | (xc,yc)为目标区域中心坐标，w为目标区域宽度，h为目标区域高度  |  

常见的目标检测数据集如Pascal VOC和COCO，采用的是第一种 `x1,y1,x2,y2` 表示物体的bounding box.  

### 准备训练数据  
PaddleDetection默认支持[COCO](http://cocodataset.org)和[Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) 和[WIDER-FACE](http://shuoyang1213.me/WIDERFACE/) 数据源。  
同时还支持自定义数据源，包括：  

(1) 自定义数据数据转换成VOC数据；  
(2) 自定义数据数据转换成COCO数据；  
(3) 自定义新的数据源，增加自定义的reader。


首先进入到`PaddleDetection`根目录下
```
cd PaddleDetection/
ppdet_root=$(pwd)
```

#### VOC数据数据  
VOC数据是[Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) 比赛使用的数据。Pascal VOC比赛不仅包含图像分类分类任务，还包含图像目标检测、图像分割等任务，其标注文件中包含多个任务的标注内容。
VOC数据集指的是Pascal VOC比赛使用的数据。用户自定义的VOC数据，xml文件中的非必须字段，请根据实际情况选择是否标注或是否使用默认值。
##### VOC数据集下载  

- 通过代码自动化下载VOC数据集  

    ```
    # 执行代码自动化下载VOC数据集  
    python dataset/voc/download_voc.py
    ```

    代码执行完成后VOC数据集文件组织结构为：
    ```
    >>cd dataset/voc/
    >>tree
    ├── create_list.py
    ├── download_voc.py
    ├── generic_det_label_list.txt
    ├── generic_det_label_list_zh.txt
    ├── label_list.txt
    ├── VOCdevkit/VOC2007
    │   ├── annotations
    │       ├── 001789.xml
    │       |   ...
    │   ├── JPEGImages
    │       ├── 001789.jpg
    │       |   ...
    │   ├── ImageSets
    │       |   ...
    ├── VOCdevkit/VOC2012
    │   ├── Annotations
    │       ├── 2011_003876.xml
    │       |   ...
    │   ├── JPEGImages
    │       ├── 2011_003876.jpg
    │       |   ...
    │   ├── ImageSets
    │       |   ...
    |   ...
    ```

    各个文件说明
    ```
    # label_list.txt 是类别名称列表，文件名必须是 label_list.txt。若使用VOC数据集，config文件中use_default_label为true时不需要这个文件
    >>cat label_list.txt
    aeroplane
    bicycle
    ...

    # trainval.txt 是训练数据集文件列表
    >>cat trainval.txt
    VOCdevkit/VOC2007/JPEGImages/007276.jpg VOCdevkit/VOC2007/Annotations/007276.xml
    VOCdevkit/VOC2012/JPEGImages/2011_002612.jpg VOCdevkit/VOC2012/Annotations/2011_002612.xml
    ...

    # test.txt 是测试数据集文件列表
    >>cat test.txt
    VOCdevkit/VOC2007/JPEGImages/000001.jpg VOCdevkit/VOC2007/Annotations/000001.xml
    ...

    # label_list.txt voc 类别名称列表
    >>cat label_list.txt

    aeroplane
    bicycle
    ...
    ```
- 已下载VOC数据集  
    按照如上数据文件组织结构组织文件即可。

##### VOC数据标注文件介绍  
VOC数据是每个图像文件对应一个同名的xml文件，xml文件中标记物体框的坐标和类别等信息。例如图像`2007_002055.jpg`：
![](../images/2007_002055.jpg)

图片对应的xml文件内包含对应图片的基本信息，比如文件名、来源、图像尺寸以及图像中包含的物体区域信息和类别信息等。

xml文件中包含以下字段：
- filename，表示图像名称。
- size，表示图像尺寸。包括：图像宽度、图像高度、图像深度。
    ```
    <size>
        <width>500</width>
        <height>375</height>
        <depth>3</depth>
    </size>
    ```
- object字段，表示每个物体。包括:

    |    标签    |    说明    |
    | :--------: | :-----------: |
    |   name    |     物体类别名称       |  
    |   pose    |    关于目标物体姿态描述（非必须字段）  |  
    |   truncated    |   如果物体的遮挡超过15-20％并且位于边界框之外，请标记为`truncated`（非必须字段）    |  
    |   difficult    |   难以识别的物体标记为`difficult`（非必须字段）      |  
    |   bndbox子标签    |  (xmin,ymin) 左上角坐标，(xmax,ymax) 右下角坐标，  |  


#### COCO数据  
COCO数据是[COCO](http://cocodataset.org) 比赛使用的数据。同样的，COCO比赛数也包含多个比赛任务，其标注文件中包含多个任务的标注内容。
COCO数据集指的是COCO比赛使用的数据。用户自定义的COCO数据，json文件中的一些字段，请根据实际情况选择是否标注或是否使用默认值。


##### COCO数据下载  
- 通过代码自动化下载COCO数据集  

    ```
    # 执行代码自动化下载COCO数据集  
    python dataset/voc/download_coco.py
    ```

    代码执行完成后COCO数据集文件组织结构为：
    ```
    >>cd dataset/coco/
    >>tree
    ├── annotations
    │   ├── instances_train2017.json
    │   ├── instances_val2017.json
    │   |   ...
    ├── train2017
    │   ├── 000000000009.jpg
    │   ├── 000000580008.jpg
    │   |   ...
    ├── val2017
    │   ├── 000000000139.jpg
    │   ├── 000000000285.jpg
    │   |   ...
    |   ...
    ```
- 已下载COCO数据集  
    按照如上数据文件组织结构组织文件即可。  

##### COCO数据标注介绍  
COCO数据标注是将所有训练图像的标注都存放到一个json文件中。数据以字典嵌套的形式存放。

json文件中包含以下key：  
- info，表示标注文件info。
- licenses，表示标注文件licenses。
- images，表示标注文件中图像信息列表，每个元素是一张图像的信息。如下为其中一张图像的信息：
    ```
    {
        'license': 3,                       # license
        'file_name': '000000391895.jpg',    # file_name
         # coco_url
        'coco_url': 'http://images.cocodataset.org/train2017/000000391895.jpg',
        'height': 360,                      # image height
        'width': 640,                       # image width
        'date_captured': '2013-11-14 11:18:45', # date_captured
        # flickr_url
        'flickr_url': 'http://farm9.staticflickr.com/8186/8119368305_4e622c8349_z.jpg',
        'id': 391895                        # image id
    }
    ```
- annotations，表示标注文件中目标物体的标注信息列表，每个元素是一个目标物体的标注信息。如下为其中一个目标物体的标注信息：
    ```
    {

        'segmentation':             # 物体的分割标注
        'area': 2765.1486500000005, # 物体的区域面积
        'iscrowd': 0,               # iscrowd
        'image_id': 558840,         # image id
        'bbox': [199.84, 200.46, 77.71, 70.88], # bbox
        'category_id': 58,          # category_id
        'id': 156                   # image id
    }
    ```

    ```
    # 查看COCO标注文件
    import json
    coco_anno = json.load(open('./annotations/instances_train2017.json'))

    # coco_anno.keys
    print('\nkeys:', coco_anno.keys())

    # 查看类别信息
    print('\n物体类别:', coco_anno['categories'])

    # 查看一共多少张图
    print('\n图像数量：', len(coco_anno['images']))

    # 查看一共多少个目标物体
    print('\n标注物体数量：', len(coco_anno['annotations']))

    # 查看一条目标物体标注信息
    print('\n查看一条目标物体标注信息：', coco_anno['annotations'][0])
    ```

    COCO数据准备如下。  
    `dataset/coco/`最初文件组织结构
    ```
    >>cd dataset/coco/
    >>tree
    ├── download_coco.py
    ```

#### 用户数据  
对于用户数据有3种处理方法：  
(1) 将用户数据转成VOC数据(根据需要仅包含物体检测所必须的标签即可)  
(2) 将用户数据转成COCO数据(根据需要仅包含物体检测所必须的标签即可)  
(3) 自定义一个用户数据的reader(较复杂数据，需要自定义reader)  

##### 用户数据转成VOC数据  
用户数据集转成VOC数据后目录结构如下（注意数据集中路径名、文件名尽量不要使用中文，避免中文编码问题导致出错）：

```
dataset/xxx/
├── annotations
│   ├── xxx1.xml
│   ├── xxx2.xml
│   ├── xxx3.xml
│   |   ...
├── images
│   ├── xxx1.jpg
│   ├── xxx2.jpg
│   ├── xxx3.jpg
│   |   ...
├── label_list.txt (必须提供，且文件名称必须是label_list.txt )
├── train.txt (训练数据集文件列表, ./images/xxx1.jpg ./annotations/xxx1.xml)
└── valid.txt (测试数据集文件列表)
```

各个文件说明
```
# label_list.txt 是类别名称列表，改文件名必须是这个
>>cat label_list.txt
classname1
classname2
...

# train.txt 是训练数据文件列表
>>cat train.txt
./images/xxx1.jpg ./annotations/xxx1.xml
./images/xxx2.jpg ./annotations/xxx2.xml
...

# valid.txt 是验证数据文件列表
>>cat valid.txt
./images/xxx3.jpg ./annotations/xxx3.xml
...
```

##### 用户数据转成COCO  
在`./tools/`中提供了`x2coco.py`用于将VOC数据集、labelme标注的数据集或cityscape数据集转换为COCO数据，例如:

（1）labelme数据转换为COCO数据：
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
（2）voc数据转换为COCO数据：
```bash
python tools/x2coco.py \
        --dataset_type voc \
        --voc_anno_dir path/to/VOCdevkit/VOC2007/Annotations/ \
        --voc_anno_list path/to/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt \
        --voc_label_list dataset/voc/label_list.txt \
        --voc_out_name voc_train.json
```

用户数据集转成COCO数据后目录结构如下（注意数据集中路径名、文件名尽量不要使用中文，避免中文编码问题导致出错）：
```
dataset/xxx/
├── annotations
│   ├── train.json  # coco数据的标注文件
│   ├── valid.json  # coco数据的标注文件
├── images
│   ├── xxx1.jpg
│   ├── xxx2.jpg
│   ├── xxx3.jpg
│   |   ...
...
```

##### 用户数据自定义reader  
如果数据集有新的数据需要添加进PaddleDetection中，您可参考数据处理文档中的[添加新数据源](../advanced_tutorials/READER.md#添加新数据源)文档部分，开发相应代码完成新的数据源支持，同时数据处理具体代码解析等可阅读[数据处理文档](../advanced_tutorials/READER.md)


#### 用户数据数据转换示例  

以[Kaggle数据集](https://www.kaggle.com/andrewmvd/road-sign-detection) 比赛数据为例，说明如何准备自定义数据。
Kaggle上的 [road-sign-detection](https://www.kaggle.com/andrewmvd/road-sign-detection) 比赛数据包含877张图像，数据类别4类：crosswalk，speedlimit，stop，trafficlight。
可从Kaggle上下载，也可以从[下载链接](https://paddlemodels.bj.bcebos.com/object_detection/roadsign.zip) 下载。
路标数据集示例图：
![](../images/road554.png)

```
# 下载解压数据
>>cd $(ppdet_root)/dataset
# 下载kaggle数据集并解压，当前文件组织结构如下

├── annotations
│   ├── road0.xml
│   ├── road1.xml
│   ├── road10.xml
│   |   ...
├── images
│   ├── road0.jpg
│   ├── road1.jpg
│   ├── road2.jpg
│   |   ...
```

将数据划分为训练集和测试集
```
# 生成 label_list.txt 文件
>>echo "speedlimit\ncrosswalk\ntrafficlight\nstop" > label_list.txt

# 生成 train.txt、valid.txt和test.txt列表文件
>>ls images/*.png | shuf > all_image_list.txt
>>awk -F"/" '{print $2}' all_image_list.txt | awk -F".png" '{print $1}'  | awk -F"\t" '{print "images/"$1".png annotations/"$1".xml"}' > all_list.txt

# 训练集、验证集、测试集比例分别约80%、10%、10%。
>>head -n 88 all_list.txt > test.txt
>>head -n 176 all_list.txt | tail -n 88 > valid.txt
>>tail -n 701 all_list.txt > train.txt

# 删除不用文件
>>rm -rf all_image_list.txt all_list.txt

最终数据集文件组织结构为：

├── annotations
│   ├── road0.xml
│   ├── road1.xml
│   ├── road10.xml
│   |   ...
├── images
│   ├── road0.jpg
│   ├── road1.jpg
│   ├── road2.jpg
│   |   ...
├── label_list.txt
├── test.txt
├── train.txt
└── valid.txt

# label_list.txt 是类别名称列表，文件名必须是 label_list.txt
>>cat label_list.txt
crosswalk
speedlimit
stop
trafficlight

# train.txt 是训练数据集文件列表，每一行是一张图像路径和对应标注文件路径，以空格分开。注意这里的路径是数据集文件夹内的相对路径。
>>cat train.txt
./images/road839.png ./annotations/road839.xml
./images/road363.png ./annotations/road363.xml
...

# valid.txt 是验证数据集文件列表，每一行是一张图像路径和对应标注文件路径，以空格分开。注意这里的路径是数据集文件夹内的相对路径。
>>cat valid.txt
./images/road218.png ./annotations/road218.xml
./images/road681.png ./annotations/road681.xml
```

也可以下载准备好的数据[下载链接](https://paddlemodels.bj.bcebos.com/object_detection/roadsign_voc.zip) ，解压到`dataset/roadsign_voc/`文件夹下即可。  
准备好数据后，一般的我们要对数据有所了解，比如图像量，图像尺寸，每一类目标区域个数，目标区域大小等。如有必要，还要对数据进行清洗。  
roadsign数据集统计:

|    数据    |    图片数量    |
| :--------: | :-----------: |
|   train    |     701       |  
|   valid    |     176       |  

**说明：**（1）用户数据，建议在训练前仔细检查数据，避免因数据标注格式错误或图像数据不完整造成训练过程中的crash  
（2）如果图像尺寸太大的话，在不限制读入数据尺寸情况下，占用内存较多，会造成内存/显存溢出，请合理设置batch_size，可从小到大尝试  
