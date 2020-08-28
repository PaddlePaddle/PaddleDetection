# 如何准备训练数据


## 一、目标检测数据格式说明
目标检测的数据格式比分类复杂，一张图像中，需要标记出各个目标区域的位置和类别。

一般的目标区域位置用一个矩形框来表示，一般用以下3种方式表达：

|         表达方式    |                 说明               |
| :----------------: | :--------------------------------: |
|     x1,y1,x2,y2    | (x1,y1)为左上角坐标，(x2,y2)为右下角坐标  |  
|       x,y,w,h      | (x,y)为左上角坐标，w为目标区域宽度，h为目标区域高度  |
|     xc,yc,w,h    | (xc,yc)为目标区域中心坐标，w为目标区域宽度，h为目标区域高度  |  

常见的目标检测数据集如VOC和COCO，采用的是第一种 `x1,y1,x2,y2` 表示物体的bounding box.  

## 二、准备训练数据方式
PaddleDetection默认支持[COCO](http://cocodataset.org)和[Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) 和[WIDER-FACE](http://shuoyang1213.me/WIDERFACE/) 数据源。  
同时还支持自定义数据源，包括：  

(1)自定义数据源转换成VOC格式；  
(2)自定义数据源转换成COCO格式；  
(3)自定义新的数据源，增加自定义的reader。


首先进入到`PaddleDetection`根目录下
```
cd PaddleDetection/
ppdet_root=$(pwd)
```

- VOC格式
    VOC格式的数据是指，每个图像文件对应一个xml文件，xml文件中标记物体框的坐标和类别等信息。  
    VOC格式是[Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) 比赛的数据格式。Pascal VOC 比赛  
    自定义的VOC格式数据，可根据需要仅标记物体检测必须使用的标签。  
    Pascal VOC数据准备如下。  
    `dataset/voc/`最初文件组织结构
    ```
    cd dataset/voc/
    tree

    ├── create_list.py
    ├── download_voc.py
    ├── generic_det_label_list.txt
    ├── generic_det_label_list_zh.txt
    ├── label_list.txt (optional)
    ```

    执行代码自动化下载VOC数据集  
    ```
    python dataset/voc/download_voc.py
    ```

    如果有已下载好的VOC数据集，将`VOCtest_06-Nov-2007.tar VOCtrainval_06-Nov-2007.tar VOCtrainval_11-May-2012.tar` 拷贝到`dataset/voc/`文件夹下
    ```
    cp VOCtest_06-Nov-2007.tar VOCtrainval_06-Nov-2007.tar VOCtrainval_11-May-2012.tar $ppdet_root/dataset/voc/
    ```
    此时文件目录结构如下：
    ```
    cd dataset/voc/
    tree

    ├── create_list.py
    ├── download_voc.py
    ├── generic_det_label_list.txt
    ├── generic_det_label_list_zh.txt
    ├── label_list.txt (optional)
    ├── VOCtest_06-Nov-2007.tar
    ├── VOCtrainval_06-Nov-2007.tar
    ├── VOCtrainval_11-May-2012.tar
    ```

    执行`dataset/voc/download_voc.py`可以省略掉下载数据时间，在`dataset/voc/`文件夹下生成 `trainval.txt` 和 `test.txt` 列表文件  
    ```
    python dataset/voc/download_voc.py
    ```
    最终VOC数据集文件组织结构为：
    ```
    cd dataset/voc/
    tree

    ├── create_list.py
    ├── download_voc.py
    ├── generic_det_label_list.txt
    ├── generic_det_label_list_zh.txt
    ├── label_list.txt (可选，配置文件中 use_default_label=true 可不提供该文件)
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

    # 各个文件说明
    # label_list.txt 是类别名称列表，文件名必须是 label_list.txt。若使用VOC数据集，config文件中use_default_label为true时不需要这个文件
    cat label_list.txt

    aeroplane
    bicycle
    ...

    # trainval.txt 是训练数据集文件列表
    cat trainval.txt

    VOCdevkit/VOC2007/JPEGImages/007276.jpg VOCdevkit/VOC2007/Annotations/007276.xml
    VOCdevkit/VOC2012/JPEGImages/2011_002612.jpg VOCdevkit/VOC2012/Annotations/2011_002612.xml
    ...

    # test.txt 是测试数据集文件列表
    cat test.txt

    VOCdevkit/VOC2007/JPEGImages/000001.jpg VOCdevkit/VOC2007/Annotations/000001.xml
    ...

    # generic_det_label_list.txt 为英文通用类别名称（将open image v5 数据集和 objects365 数据集合并到一起  ）
    cat generic_det_label_list.txt

    Infant bed
    Rose
    # generic_det_label_list_zh.txt 为中文通用类别名称（将open image v5 数据集和 objects365 数据集合并到一起  ）
    cat generic_det_label_list.txt

    婴儿床
    玫瑰
    ```

    如果有已下载好且解压好的VOC数据集，按照如上文件组织结构拷贝文件即可。  
    **说明： 如果你在yaml配置文件中设置use_default_label=False, 将从label_list.txt 中读取类别列表，反之则可以没有label_list.txt文件，检测库会使用Pascal VOC数据集的默 认类别列表，默认类别列表定义在voc.py**

- COCO  
    COCO格式的数据是指，所有图像数据的标签按照COCO的格式放到一个json文件中。  
    [COCO](http://cocodataset.org) 比赛数据请参考官方网站[http://cocodataset.org](http://cocodataset.org) 。  
    自定义的COCO格式数据，可根据需要仅标记物体检测必须使用的标签。  
    COCO数据准备如下。  
    `dataset/coco/`最初文件组织结构
    ```
    cd dataset/coco/
    tree

    ├── download_coco.py.py
    ```

    执行代码自动化下载COCO数据集  
    ```
    python dataset/voc/download_coco.py
    ```

    如果有已下载好的COCO数据集，分别解压`train2017.zip val2017.zip annotations_trainval2017.zip annotations_trainval2014.zip`文件，并按照如下文件结构组织文件即可：
    ```
    cd dataset/coco/
    tree
    ├── annotations
    │   ├── instances_train2014.json
    │   ├── instances_train2017.json
    │   ├── instances_val2014.json
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
    **注意：** 在`./ppdet/data/tools/x2coco.py`用于将labelme标注的数据集或cityscape数据集转换为COCO数据集。详细可参考文档[自定义数据集](Custom_DataSet.md)  


- 用户数据  
    用户数据有3种处理方法：  
    (1)将用户数据转成VOC格式(根据需要仅包含物体检测所必须的标签即可)  
    (2)将用户数据转成COCO格式(根据需要仅包含物体检测所必须的标签即可)  
    (3)自定义一个用户数据的reader(较复杂数据格式，需要自定义reader)  

    ```
    cd dataset/xxx
    ```
    用户数据集转成VOC格式后目录结构如下（注意数据集中路径名、文件名尽量不要使用中文，避免中文编码问题导致出错）：

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

    # label_list.txt 是类别名称列表，改文件名必须是这个
    cat label_list.txt
    classname1
    classname2
    ...

    # train.txt 是训练数据文件列表
    cat train.txt
    ./images/xxx1.jpg ./annotations/xxx1.xml
    ./images/xxx2.jpg ./annotations/xxx2.xml
    ...

    # valid.txt 是验证数据文件列表
    cat valid.txt
    ./images/xxx3.jpg ./annotations/xxx3.xml
    ...
    ```

    以[Kaggle数据集](https://www.kaggle.com/andrewmvd/road-sign-detection) 比赛数据为例，说明如何准备自定义数据。
    Kaggle上的 [road-sign-detection](https://www.kaggle.com/andrewmvd/road-sign-detection) 比赛数据包含877张图像，数据类别4类：crosswalk，speedlimit，stop，trafficlight。
    可从Kaggle上下载，也可以从[下载链接](https://paddlemodels.bj.bcebos.com/object_detection/roadsign.zip) 下载。
    路标数据集示例图：
    ![](../images/road554.png)

    ```
    # 下载解压数据
    cd $(ppdet_root)/dataset
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
    echo "crosswalk\nspeedlimit\nstop\ntrafficlight" > label_list.txt

    # 生成 train.txt、valid.txt和test.txt列表文件
    ls images/*.png | shuf > all_image_list.txt
    awk -F"/" '{print $2}' all_image_list.txt | awk -F".png" '{print $1}'  | awk -F"\t" '{print "images/"$1".png annotations/"$1".xml"}' > all_list.txt

    # 训练集、验证集、测试集比例分别为80%、10%、10%。
    head -n 88 all_list.txt > test.txt
    head -n 176 all_list.txt | tail -n 88 > valid.txt
    tail -n 701 all_list.txt > train.txt

    # 删除不用文件
    rm -rf all_image_list.txt all_list.txt

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
    cat label_list.txt

    crosswalk
    speedlimit
    stop
    trafficlight

    # train.txt 是训练数据集文件列表
    cat train.txt

    ./images/road839.png ./annotations/road839.xml
    ./images/road363.png ./annotations/road363.xml
    ...

    # valid.txt 是验证数据集文件列表
    cat valid.txt

    ./images/road218.png ./annotations/road218.xml
    ./images/road681.png ./annotations/road681.xml
    ```

    也可以下载准备好的数据，([下载链接](https://paddlemodels.bj.bcebos.com/object_detection/roadsign_voc.zip) ，解压到`dataset/`文件夹下重命名为`roadsign`即可。  
    准备好数据后，一般的我们要对数据有所了解，比如图像量，图像尺寸，每一类目标区域个数，目标区域大小等。如有必要，还要对数据进行清洗。  
    roadsign数据集统计:

    |    数据    |    图片数量    |
    | :--------: | :-----------: |
    |   train    |     701       |  
    |   valid    |     176       |  

    **说明：**（1）用户数据，建议在训练前仔细检查数据，避免因数据标注格式错误或图像数据不完整造成训练过程中的crash  
    （2）如果图像尺寸太大的话，在不限制读入数据尺寸情况下，占用内存较多，会造成内存/显存溢出，请合理设置batch_size，可从小到大尝试  
