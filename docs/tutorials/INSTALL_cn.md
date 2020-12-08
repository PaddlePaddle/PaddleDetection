# 安装说明

---
## 目录

- [简介](#简介)
- [安装PaddlePaddle](#安装PaddlePaddle)
- [其他依赖安装](#其他依赖安装)
- [PaddleDetection](#PaddleDetection)


## 简介

这份文档介绍了如何安装PaddleDetection及其依赖项(包括PaddlePaddle)。

PaddleDetection的相关信息，请参考[README.md](https://github.com/PaddlePaddle/PaddleDetection/blob/master/README.md).


## 安装PaddlePaddle

**环境需求:**

- OS 64位操作系统
- Python 3(3.5.1+/3.6/3.7)，64位版本
- pip/pip3(9.0.1+)，64位版本操作系统是
- CUDA >= 9.0
- cuDNN >= 7.6

如果需要 GPU 多卡训练，请先安装NCCL。


## 其他依赖安装

[COCO-API](https://github.com/cocodataset/cocoapi):

运行需要COCO-API，安装方式如下：

    # 安装pycocotools
    pip install pycocotools

**windows用户安装COCO-API方式：**

    # 若Cython未安装，请安装Cython
    pip install Cython

    # 由于原版cocoapi不支持windows，采用第三方实现版本，该版本仅支持Python3
    pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI

## PaddleDetection

**安装Python依赖库：**

Python依赖库在[requirements.txt](https://github.com/PaddlePaddle/PaddleDetection/blob/master/requirements.txt)中给出，可通过如下命令安装：

```
pip install -r requirements.txt
```
**注意：`llvmlite`需要安装`0.33`版本，`numba`需要安装`0.50`版本**


**克隆PaddleDetection库：**

您可以通过以下命令克隆PaddleDetection：

```
cd <path/to/clone/PaddleDetection>
git clone https://github.com/PaddlePaddle/PaddleDetection.git
```

也可以通过 [https://gitee.com/paddlepaddle/PaddleDetection](https://gitee.com/paddlepaddle/PaddleDetection) 克隆。
```
cd <path/to/clone/PazcccccccccccddleDetection>
git clone https://gitee.com/paddlepaddle/PaddleDetection
```
