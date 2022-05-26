# 安装说明

---
## 目录

- [简介](#简介)
- [安装PaddlePaddle](#安装PaddlePaddle)
- [其他依赖安装](#其他依赖安装)
- [PaddleDetection](#PaddleDetection)


## 简介

这份文档介绍了如何安装PaddleDetection及其依赖项(包括PaddlePaddle)。

PaddleDetection的相关信息，请参考[README.md](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/README_cn.md).


## 安装PaddlePaddle

**环境需求:**

- OS 64位操作系统
- Python2 >= 2.7.15 or Python 3(3.5.1+/3.6/3.7)，64位版本
- pip/pip3(9.0.1+)，64位版本操作系统
- CUDA >= 9.0
- cuDNN >= 7.6

如果需要 GPU 多卡训练，请先安装NCCL(Windows暂不支持nccl)。

PaddleDetection 依赖 PaddlePaddle 版本关系：

|  PaddleDetection版本  | PaddlePaddle版本  |    备注    |
| :------------------: | :---------------: | :-------: |
|    release/0.3       |        >=1.7      |     --    |
|    release/0.4       |       >= 1.8.4    |  PP-YOLO依赖1.8.4 |
|    release/0.5       |       >= 1.8.4    |  大部分模型>=1.8.4即可运行，Cascade R-CNN系列模型与SOLOv2依赖2.0.0.rc版本 |
|    release/2.0-rc    |       >= 2.0.1    |     --    |


```
# install paddlepaddle
# 如果您的机器安装的是CUDA9，请运行以下命令安装
python -m pip install paddlepaddle-gpu==2.0.1.post90 -i https://mirror.baidu.com/pypi/simple

如果您的机器安装的是CUDA10.1，请运行以下命令安装
python -m pip install paddlepaddle-gpu==2.0.1.post101 -f https://paddlepaddle.org.cn/whl/mkl/stable.html

如果您的机器是CPU，请运行以下命令安装
python -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
```

更多的安装方式如conda, docker安装，请参考[安装文档](https://www.paddlepaddle.org.cn/install/quick)中的说明进行操作

请确保您的PaddlePaddle安装成功并且版本不低于需求版本。使用以下命令进行验证。

```
# 在您的Python解释器中确认PaddlePaddle安装成功
>>> import paddle
>>> paddle.utils.run_check()

# 确认PaddlePaddle版本
python -c "import paddle; print(paddle.__version__)"
```


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

Python依赖库在[requirements.txt](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/requirements.txt) 中给出，可通过如下命令安装：

```
pip install -r requirements.txt
```

**克隆PaddleDetection库：**

您可以通过以下命令克隆PaddleDetection：

```
cd <path/to/clone/PaddleDetection>
git clone https://github.com/PaddlePaddle/PaddleDetection.git
```
**提示：**

也可以通过 [https://gitee.com/paddlepaddle/PaddleDetection](https://gitee.com/paddlepaddle/PaddleDetection) 克隆。
```
cd <path/to/clone/PaddleDetection>
git clone https://gitee.com/paddlepaddle/PaddleDetection
```

**确认测试通过：**

```
python ppdet/modeling/tests/test_architectures.py
```

测试通过后会提示如下信息：
```
..........
----------------------------------------------------------------------
Ran 12 tests in 2.480s
OK (skipped=2)
```

**预训练模型预测**

使用预训练模型预测图像，快速体验模型预测效果：

```
# use_gpu参数设置是否使用GPU
python tools/infer.py -c configs/ppyolo/ppyolo.yml -o use_gpu=true weights=https://paddlemodels.bj.bcebos.com/object_detection/ppyolo.pdparams --infer_img=demo/000000014439.jpg
```

会在`output`文件夹下生成一个画有预测结果的同名图像。

结果如下图：

![](../images/000000014439.jpg)
