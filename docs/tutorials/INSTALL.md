English | [简体中文](INSTALL_cn.md)

# Installation


This document covers how to install PaddleDetection and its dependencies
(including PaddlePaddle), together with COCO and Pascal VOC dataset.

For general information about PaddleDetection, please see [README.md](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/).

## Requirements:

- PaddlePaddle 2.0.1
- OS 64 bit
- Python 3(3.5.1+/3.6/3.7)，64 bit
- pip/pip3(9.0.1+), 64 bit
- CUDA >= 9.0
- cuDNN >= 7.6


## Instruction

### 1. Install PaddlePaddle

```
# CUDA9.0
python -m pip install paddlepaddle-gpu==2.0.1.post90 -i https://mirror.baidu.com/pypi/simple

# CUDA10.1
python -m pip install paddlepaddle-gpu==2.0.1.post101 -f https://mirror.baidu.com/pypi/simple

# CPU
python -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
```

For more installation methods such as conda or compile with source code, please refer to the [installation document](https://www.paddlepaddle.org.cn/documentation/docs/en/install/index_en.html)

Please make sure that your PaddlePaddle is installed successfully and the version is not lower than the required version. Use the following command to verify.

```
# check
>>> import paddle
>>> paddle.utils.run_check()

# confirm the paddle's version
python -c "import paddle; print(paddle.__version__)"
```

**Note**

1.  If you want to use PaddleDetection on multi-GPU, please install NCCL at first.


### 2. Install PaddleDetection

PaddleDetection can be installed in the following two ways:

#### 2.1 Install via pip

**Note:** Installing via pip only supports Python3

```
# install paddledet via pip
pip install paddledet==2.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

# Download and use the configuration files and code examples in the source code
git clone https://github.com/PaddlePaddle/PaddleDetection.git
cd PaddleDetection
```

#### 2.2 Compile and install from Source code

```
# Clone PaddleDetection repository
cd <path/to/clone/PaddleDetection>
git clone https://github.com/PaddlePaddle/PaddleDetection.git

# Install other dependencies
pip install -r requirements.txt

# Install PaddleDetection
cd PaddleDetection
python setup.py install
```

**Note**

1. Because the origin version of cocoapi does not support windows, another version is used which only supports Python3:

    ```pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI```

After installation, make sure the tests pass:

```shell
python ppdet/modeling/tests/test_architectures.py
```

If the tests are passed, the following information will be prompted:

```
..........
----------------------------------------------------------------------
Ran 12 tests in 2.480s
OK (skipped=2)
```

## Inference demo

**Congratulation!** Now you have installed PaddleDetection successfully and try our inference demo:

```
# Predict an image by GPU
export CUDA_VISIBLE_DEVICES=0
python tools/infer.py -c configs/ppyolo/ppyolo.yml -o use_gpu=true weights=https://paddlemodels.bj.bcebos.com/object_detection/ppyolo.pdparams --infer_img=demo/000000014439.jpg
```

An image of the same name with the predicted result will be generated under the `output` folder.
The result is as shown below：

![](../images/000000014439.jpg)
