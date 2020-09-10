# 安装说明

---
## 目录

- [简介](#简介)
- [安装PaddlePaddle](#paddlepaddle)
- [其他依赖安装](#其他依赖安装)
- [PaddleDetection](#PaddleDetection)


## 简介

这份文档介绍了如何安装PaddleDetection及其依赖项(包括PaddlePaddle)。

PaddleDetection的相关信息，请参考[README.md](https://github.com/PaddlePaddle/PaddleDetection/blob/master/README.md).


## 安装PaddlePaddle

PaddleDetection 依赖 PaddlePaddle 版本关系：

**PaddleDetection v0.4需要PaddlePaddle>=1.8.4**

PaddlePaddle安装请按照[安装文档](http://www.paddlepaddle.org.cn/)中的说明进行操作。

请确保您的PaddlePaddle安装成功并且版本不低于需求版本。使用以下命令进行验证。

```
# 在您的Python解释器中确认PaddlePaddle安装成功
>>> import paddle.fluid as fluid
>>> fluid.install_check.run_check()

# 确认PaddlePaddle版本
python -c "import paddle; print(paddle.__version__)"
```

**环境需求:**

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
    # 或者使用pip安装
    pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"

**windows用户安装COCO-API方式：**

    # 若Cython未安装，请安装Cython
    pip install Cython
    # 由于原版cocoapi不支持windows，采用第三方实现版本，该版本仅支持Python3
    pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI

## PaddleDetection

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

**安装Python依赖库：**

Python依赖库在[requirements.txt](https://github.com/PaddlePaddle/PaddleDetection/blob/master/requirements.txt)中给出，可通过如下命令安装：

```
pip install -r requirements.txt
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
python tools/infer.py -c configs/ppyolo/ppyolo.yml -o use_gpu=true weights=https://paddlemodels.bj.bcebos.com/object_detection/ppyolo.pdparams --infer_img=demo/000000014439_640x640.jpg
```

会在`output`文件夹下生成一个画有预测结果的同名图像。

结果如下图：

![](../images/000000014439_640x640.jpg)
