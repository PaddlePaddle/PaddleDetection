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

It is recommened to install PaddleDetection and begin your object detection journey via docker environment. Please follow the instruction below and if you want to use your local environment, you could skip step 1.


### 1. (Recommended) Prepare docker environment

For example, the environment is CUDA10.1 and CUDNN 7.6

```bash
# Firstly, pull the PaddlePaddle image
sudo docker pull paddlepaddle/paddle:latest-dev-cuda10.1-cudnn7-gcc82

# Switch to the working directory
cd /home/work

# Create a container called ppdet and
# mount the current directory which may contains the dataset
# to /paddle directory in the container
sudo nvidia-docker run --name ppdet -v $PWD:/paddle --privileged --shm-size=4G --network=host -it paddlepaddle/paddle:latest-dev-cuda10.1-cudnn7-gcc82 /bin/bash
```

You can see [DockerHub](https://hub.docker.com/r/paddlepaddle/paddle/tags/) to get the image that matches your machine.

```
# ctrl+P+Q to exit docker, to re-enter docker using the following command:
sudo docker exec -it ppdet /bin/bash
```

For more docker usage, please refer to the PaddlePaddle [document](https://www.paddlepaddle.org.cn/documentation/docs/en/install/docker/fromdocker_en.html).

### 2. Install PaddlePaddle

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


### 3. Install PaddleDetection


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
