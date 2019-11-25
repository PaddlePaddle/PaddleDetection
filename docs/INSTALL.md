English | [简体中文](INSTALL_cn.md)

# Installation

---
## Table of Contents

- [Introduction](#introduction)
- [PaddlePaddle](#paddlepaddle)
- [Other Dependencies](#other-dependencies)
- [PaddleDetection](#paddle-detection)
- [Datasets](#datasets)


## Introduction

This document covers how to install PaddleDetection, its dependencies
(including PaddlePaddle), together with COCO and Pascal VOC dataset.

For general information about PaddleDetection, please see [README.md](../README.md).


## PaddlePaddle

Running PaddleDetection requires PaddlePaddle Fluid v.1.6 and later. please follow the instructions in [installation document](http://www.paddlepaddle.org.cn/).

Please make sure your PaddlePaddle installation was successful and the version
of your PaddlePaddle is not lower than required. Verify with the following commands.

```
# To check PaddlePaddle installation in your Python interpreter
>>> import paddle.fluid as fluid
>>> fluid.install_check.run_check()

# To check PaddlePaddle version
python -c "import paddle; print(paddle.__version__)"
```

### Requirements:

- Python2 or Python3 (Only support Python3 for windows)
- CUDA >= 8.0
- cuDNN >= 5.0
- nccl >= 2.1.2


## Other Dependencies

[COCO-API](https://github.com/cocodataset/cocoapi):

COCO-API is needed for running. Installation is as follows:

    git clone https://github.com/cocodataset/cocoapi.git
    cd cocoapi/PythonAPI
    # if cython is not installed
    pip install Cython
    # Install into global site-packages
    make install
    # Alternatively, if you do not have permissions or prefer
    # not to install the COCO API into global site-packages
    python setup.py install --user

**Installation of COCO-API in windows:**

    # if cython is not installed
    pip install Cython
    # Because the origin version of cocoapi does not support windows, another version is used which only supports Python3
    pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI

## PaddleDetection

**Clone Paddle models repository:**

You can clone PaddleDetection with the following commands:

```
cd <path/to/clone/PaddleDetection>
git clone https://github.com/PaddlePaddle/PaddleDetection.git
```

**Install Python dependencies:**

Required python packages are specified in [requirements.txt](../requirements.txt), and can be installed with:

```
pip install -r requirements.txt
```

**Make sure the tests pass:**

```
export PYTHONPATH=`pwd`:$PYTHONPATH
python ppdet/modeling/tests/test_architectures.py
```


## Datasets

PaddleDetection includes support for [COCO](http://cocodataset.org) and [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) by default, please follow these instructions to set up the dataset.

**Create symlinks for local datasets:**

Default dataset path in config files is `dataset/coco` and `dataset/voc`, if the
datasets are already available on disk, you can simply create symlinks to
their directories:

```
ln -sf <path/to/coco> <path/to/paddle_detection>/dataset/coco
ln -sf <path/to/voc> <path/to/paddle_detection>/dataset/voc
```

For Pascal VOC dataset, you should create file list by:

```
export PYTHONPATH=$PYTHONPATH:.
python dataset/voc/create_list.py
```

**Download datasets manually:**

On the other hand, to download the datasets, run the following commands:

- COCO

```
export PYTHONPATH=$PYTHONPATH:.
python dataset/coco/download_coco.py
```

`COCO` dataset with directory structures like this:

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

`Pascal VOC` dataset with directory structure like this:

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

**NOTE:** If you set `use_default_label=False` in yaml configs, the `label_list.txt`
of Pascal VOC dataset will be read, otherwise, `label_list.txt` is unnecessary and
the default Pascal VOC label list which defined in
[voc\_loader.py](../ppdet/data/source/voc_loader.py) will be used.

**Download datasets automatically:**

If a training session is started but the dataset is not setup properly (e.g,
not found in `dataset/coco` or `dataset/voc`), PaddleDetection can automatically
download them from [COCO-2017](http://images.cocodataset.org) and
[VOC2012](http://host.robots.ox.ac.uk/pascal/VOC), the decompressed datasets
will be cached in `~/.cache/paddle/dataset/` and can be discovered automatically
subsequently.


**NOTE:** For further informations on the datasets, please see [DATA.md](DATA.md)
