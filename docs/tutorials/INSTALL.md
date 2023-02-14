English | [简体中文](INSTALL_cn.md)

# Installation


This document covers how to install PaddleDetection and its dependencies
(including PaddlePaddle), together with COCO and Pascal VOC dataset.

For general information about PaddleDetection, please see [README.md](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6).

## Requirements:

- PaddlePaddle 2.2
- OS 64 bit
- Python 3(3.5.1+/3.6/3.7/3.8/3.9/3.10)，64 bit
- pip/pip3(9.0.1+), 64 bit
- CUDA >= 10.2
- cuDNN >= 7.6


Dependency of PaddleDetection and PaddlePaddle:

| PaddleDetection version | PaddlePaddle version  |    tips    |
| :----------------: | :---------------: | :-------: |
|    develop           |       >= 2.3.2   |     Dygraph mode is set as default    |
|    release/2.6       |       >= 2.3.2   |     Dygraph mode is set as default    |
|    release/2.5       |       >= 2.2.2   |     Dygraph mode is set as default    |
|    release/2.4       |       >= 2.2.2   |     Dygraph mode is set as default    |
|    release/2.3       |       >= 2.2.0rc |     Dygraph mode is set as default    |
|    release/2.2       |       >= 2.1.2   |     Dygraph mode is set as default    |
|    release/2.1       |       >= 2.1.0   |     Dygraph mode is set as default    |
|    release/2.0       |       >= 2.0.1    |     Dygraph mode is set as default    |
|    release/2.0-rc    |       >= 2.0.1    |     --    |
|    release/0.5       |       >= 1.8.4    |  Cascade R-CNN and SOLOv2 depends on 2.0.0.rc |
|    release/0.4       |       >= 1.8.4    |  PP-YOLO depends on 1.8.4 |
|    release/0.3       |        >=1.7      |     --    |


## Instruction

### 1. Install PaddlePaddle

```

# CUDA10.2
python -m pip install paddlepaddle-gpu==2.3.2 -i https://mirror.baidu.com/pypi/simple

# CPU
python -m pip install paddlepaddle==2.3.2 -i https://mirror.baidu.com/pypi/simple
```

- For more CUDA version or environment to quick install, please refer to the [PaddlePaddle Quick Installation document](https://www.paddlepaddle.org.cn/install/quick)
- For more installation methods such as conda or compile with source code, please refer to the [installation document](https://www.paddlepaddle.org.cn/documentation/docs/en/install/index_en.html)

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



**Note:** Installing via pip only supports Python3

```

# Clone PaddleDetection repository
cd <path/to/clone/PaddleDetection>
git clone https://github.com/PaddlePaddle/PaddleDetection.git

# Install other dependencies
cd PaddleDetection
pip install -r requirements.txt

# Compile and install paddledet
python setup.py install

```

**Note**

1. If you are working on Windows OS, `pycocotools` installing may failed because of the origin version of cocoapi does not support windows, another version can be used used which only supports Python3:

    ```pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI```

2. If you are using Python <= 3.6, `pycocotools` installing may failed with error like `distutils.errors.DistutilsError: Could not find suitable distribution for Requirement.parse('cython>=0.27.3')`, please install `cython` firstly, for example `pip install cython`

After installation, make sure the tests pass:

```shell
python ppdet/modeling/tests/test_architectures.py
```

If the tests are passed, the following information will be prompted:

```
.......
----------------------------------------------------------------------
Ran 7 tests in 12.816s
OK
```

## Use built Docker images

> If you  do not have a Docker environment, please refer to [Docker](https://www.docker.com/).

We provide docker images containing the latest PaddleDetection code, and all environment and package dependencies are pre-installed. All you have to do is to **pull and run the docker image**. Then you can enjoy PaddleDetection without any extra steps.

Get these images and guidance in [docker hub](https://hub.docker.com/repository/docker/paddlecloud/paddledetection), including CPU, GPU, ROCm environment versions.

If you have some customized requirements about automatic building docker images, you can get it in github repo [PaddlePaddle/PaddleCloud](https://github.com/PaddlePaddle/PaddleCloud/tree/main/tekton).

## Inference demo

**Congratulation!** Now you have installed PaddleDetection successfully and try our inference demo:

```
# Predict an image by GPU
export CUDA_VISIBLE_DEVICES=0
python tools/infer.py -c configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml -o use_gpu=true weights=https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_1x_coco.pdparams --infer_img=demo/000000014439.jpg
```

An image of the same name with the predicted result will be generated under the `output` folder.
The result is as shown below：

![](../images/000000014439.jpg)
