English | [简体中文](INSTALL_cn.md)

# Installation

---
## Table of Contents

- [Introduction](#introduction)
- [PaddlePaddle](#paddlepaddle)
- [Other Dependencies](#other-dependencies)
- [PaddleDetection](#paddle-detection)


## Introduction

This document covers how to install PaddleDetection, its dependencies
(including PaddlePaddle), together with COCO and Pascal VOC dataset.

For general information about PaddleDetection, please see [README.md](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/).


## Install PaddlePaddle

### Requirements:
- OS 64 bit
- Python2 >= 2.7.15 or Python 3(3.5.1+/3.6/3.7)，64 bit
- pip/pip3(9.0.1+), 64 bit
- CUDA >= 9.0
- cuDNN >= 7.6

If you need GPU multi-card training, firstly please install NCCL. (Windows does not support nccl).

PaddleDetection depends on PaddlePaddle version relationship:

| PaddleDetection version | PaddlePaddle version  |    tips    |
| :----------------: | :---------------: | :-------: |
|    release/0.3       |        >=1.7      |     --    |
|    release/0.4       |       >= 1.8.4    |  PP-YOLO depends on 1.8.4 |
|    release/0.5       |       >= 1.8.4    |  Cascade R-CNN and SOLOv2 depends on 2.0.0.rc |
|    release/2.0-rc    |       >= 2.0.1    |     --    |


If you want install paddlepaddle, please follow the instructions in [installation document](http://www.paddlepaddle.org.cn/).

```
# install paddlepaddle
# install paddlepaddle CUDA9.0
python -m pip install paddlepaddle-gpu==2.0.1.post90 -i https://mirror.baidu.com/pypi/simple

install paddlepaddle CUDA10.0
python -m pip install paddlepaddle-gpu==2.0.1.post101 -f https://paddlepaddle.org.cn/whl/mkl/stable.html

install paddlepaddle CPU
python -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
```

For more installation methods such as conda, docker installation, please refer to the instructions in the [installation document](https://www.paddlepaddle.org.cn/install/quick)

Please make sure that your PaddlePaddle is installed successfully and the version is not lower than the required version. Use the following command to verify.

```
# check
>>> import paddle
>>> paddle.utils.run_check()

# confirm the paddle's version
python -c "import paddle; print(paddle.__version__)"
```


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
    # or with pip
    pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"

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

Required python packages are specified in [requirements.txt](https://github.com/PaddlePaddle/PaddleDetection/blob/master/requirements.txt), and can be installed with:

```
pip install -r requirements.txt
```

**Make sure the tests pass:**

```shell
python ppdet/modeling/tests/test_architectures.py
```

After the test is passed, the following information will be prompted:
```
..........
----------------------------------------------------------------------
Ran 12 tests in 2.480s
OK (skipped=2)
```

**Infer by pretrained-model**

Use the pre-trained model to predict the image:

```
# set use_gpu
python tools/infer.py -c configs/ppyolo/ppyolo.yml -o use_gpu=true weights=https://paddlemodels.bj.bcebos.com/object_detection/ppyolo.pdparams --infer_img=demo/000000014439.jpg
```

An image of the same name with the predicted result will be generated under the `output` folder.
The result is as shown below：

![](../images/000000014439.jpg)
