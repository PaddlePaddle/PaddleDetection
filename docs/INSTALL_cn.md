# 安装文档

---
## 目录

- [简介](#introduction)
- [PaddlePaddle](#paddlepaddle)
- [其他依赖安装](#other-dependencies)
- [PaddleDetection](#paddle-detection)
- [数据集](#datasets)


## 简介

这份文档介绍了如何安装PaddleDetection及其依赖项(包括PaddlePaddle)，以及COCO和Pascal VOC数据集。

PaddleDetection的相关信息，请参考[README.md](../README.md).


## PaddlePaddle


运行PaddleDetection需要PaddlePaddle Fluid v.1.6及更高版本。请按照[安装文档](http://www.paddlepaddle.org.cn/)中的说明进行操作。

请确保您的PaddlePaddle安装成功并且版本不低于需求版本。使用以下命令进行验证。

```
# 在您的Python解释器中确认PaddlePaddle安装成功
>>> import paddle.fluid as fluid
>>> fluid.install_check.run_check()

# 确认PaddlePaddle版本
python -c "import paddle; print(paddle.__version__)"
```

### 环境需求:

- Python2 or Python3 (windows系统仅支持Python3)
- CUDA >= 8.0
- cuDNN >= 5.0
- nccl >= 2.1.2


## 其他依赖安装

[COCO-API](https://github.com/cocodataset/cocoapi):

运行需要COCO-API，安装方式如下：

    git clone https://github.com/cocodataset/cocoapi.git
    cd cocoapi/PythonAPI
    # 若Cython未安装，请安装Cython
    pip install Cython
    # 安装至全局site-packages
    make install
    # 若您没有权限或更倾向不安装至全局site-packages
    python setup.py install --user

**windows用户安装COCO-API方式：**

    # 若Cython未安装，请安装Cython
    pip install Cython
    # 由于原版cocoapi不支持windows，采用第三方实现版本，该版本仅支持Python3
    pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI

## PaddleDetection

**克隆Paddle models模型库：**

您可以通过以下命令克隆PaddleDetection：

```
cd <path/to/clone/PaddleDetection>
git clone https://github.com/PaddlePaddle/PaddleDetection.git
```

**安装Python依赖库：**

Python依赖库在[requirements.txt](../requirements.txt)中给出，可通过如下命令安装：

```
pip install -r requirements.txt
```

**确认测试通过：**

```
export PYTHONPATH=`pwd`:$PYTHONPATH
python ppdet/modeling/tests/test_architectures.py
```


## 数据集


PaddleDetection默认支持[COCO](http://cocodataset.org)和[Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/)，
请按照如下步骤设置数据集。

**为本地数据集创建软链接：**


配置文件中默认的数据集路径是`dataset/coco`和`dataset/voc`，如果您本地磁盘上已有数据集，
只需创建软链接至数据集目录：

```
ln -sf <path/to/coco> <path/to/paddle_detection>/dataset/coco
ln -sf <path/to/voc> <path/to/paddle_detection>/dataset/voc
```

对于Pascal VOC数据集，需通过如下命令创建文件列表：

```
export PYTHONPATH=$PYTHONPATH:.
python dataset/voc/create_list.py
```

**手动下载数据集：**

若您本地没有数据集，可通过如下命令下载：

- COCO

```
export PYTHONPATH=$PYTHONPATH:.
python dataset/coco/download_coco.py
```

`COCO` 数据集目录结构如下：

  ```
  dataset/coco/
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

- Pascal VOC

```
export PYTHONPATH=$PYTHONPATH:.
python dataset/voc/download_voc.py
python dataset/voc/create_list.py
```

`Pascal VOC` 数据集目录结构如下：

  ```
  dataset/voc/
  ├── train.txt
  ├── val.txt
  ├── test.txt
  ├── label_list.txt (optional)
  ├── VOCdevkit/VOC2007
  │   ├── Annotations
  │       ├── 001789.xml
  │       |   ...
  │   ├── JPEGImages
  │       ├── 001789.xml
  │       |   ...
  │   ├── ImageSets
  │       |   ...
  ├── VOCdevkit/VOC2012
  │   ├── Annotations
  │       ├── 003876.xml
  │       |   ...
  │   ├── JPEGImages
  │       ├── 003876.xml
  │       |   ...
  │   ├── ImageSets
  │       |   ...
  |   ...
  ```

**说明：** 如果你在yaml配置文件中设置`use_default_label=False`, 将从`label_list.txt`
中读取类别列表，反之则可以没有`label_list.txt`文件，检测库会使用Pascal VOC数据集的默
认类别列表，默认类别列表定义在[voc\_loader.py](../ppdet/data/source/voc_loader.py)

**自动下载数据集：**

若您在数据集未成功设置（例如，在`dataset/coco`或`dataset/voc`中找不到）的情况下开始运行，
PaddleDetection将自动从[COCO-2017](http://images.cocodataset.org)或
[VOC2012](http://host.robots.ox.ac.uk/pascal/VOC)下载，解压后的数据集将被保存在
`〜/.cache/paddle/dataset/`目录下，下次运行时，也可自动从该目录发现数据集。


**说明：** 更多有关数据集的介绍，请参考[DATA.md](DATA_cn.md)
