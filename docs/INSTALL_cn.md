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


运行PaddleDetection需要PaddlePaddle Fluid v.1.5及更高版本。请按照[安装文档](http://www.paddlepaddle.org.cn/)中的说明进行操作。

请确保您的PaddlePaddle安装成功并且版本不低于需求版本。使用以下命令进行验证。

```
# 在您的Python解释器中确认PaddlePaddle安装成功
>>> import paddle.fluid as fluid
>>> fluid.install_check.run_check()

# 确认PaddlePaddle版本
python -c "import paddle; print(paddle.__version__)"
```

### 环境需求:

- Python2 or Python3
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


## PaddleDetection

**克隆Paddle models模型库：**

您可以通过以下命令克隆Paddle models模型库并切换工作目录至PaddleDetection：

```
cd <path/to/clone/models>
git clone https://github.com/PaddlePaddle/models
cd models/PaddleCV/PaddleDetection
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

**手动下载数据集：**

若您本地没有数据集，可通过如下命令下载：

- COCO

```
cd dataset/coco
./download.sh
```

- Pascal VOC

```
cd dataset/voc
./download.sh
```

**自动下载数据集：**

若您在数据集未成功设置（例如，在`dataset/coco`或`dataset/voc`中找不到）的情况下开始运行，
PaddleDetection将自动从[COCO-2017](http://images.cocodataset.org)或
[VOC2012](http://host.robots.ox.ac.uk/pascal/VOC)下载，解压后的数据集将被保存在
`〜/.cache/paddle/dataset/`目录下，下次运行时，也可自动从该目录发现数据集。


**说明：** 更多有关数据集的介绍，请参考[DATA.md](DATA_cn.md)
