[English](INSTALL.md) | 简体中文


# 安装文档



## 环境要求

- PaddlePaddle 2.0.1
- OS 64位操作系统
- Python 3(3.5.1+/3.6/3.7)，64位版本
- pip/pip3(9.0.1+)，64位版本
- CUDA >= 9.0
- cuDNN >= 7.6


## 安装说明

建议使用docker环境安装PaddleDetection并开启你的目标检测之旅。请按照如下步骤说明进行安装，如果您希望使用本机环境，可以跳过步骤1.

### 1. （推荐）准备docker环境

已CUDA10.1， CUDNN7.6为例

```bash
# 首先拉去PaddlePaddle镜像
sudo docker pull paddlepaddle/paddle:latest-dev-cuda10.1-cudnn7-gcc82

# 切换到工作目录
cd /home/work

# 创建ppdet容器
# 将存放数据的当前目录映射到容器中的/ppdet目录中
sudo nvidia-docker run --name ppdet -v $PWD:/paddle --privileged --shm-size=4G --network=host -it paddlepaddle/paddle:latest-dev-cuda10.1-cudnn7-gcc82 /bin/bash
```

可以在[DockerHub](https://hub.docker.com/r/paddlepaddle/paddle/tags/) 中找到匹配您机器环境的镜像

```
# ctrl+P+Q 退出容器, 使用如下命令重新进入docker环境：
sudo docker exec -it ppdet /bin/bash
```

其他更多docker用法，请参考PaddlePaddle[文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/docker/fromdocker.html)


### 安装PaddlePaddle

```
# CUDA9.0
python -m pip install paddlepaddle-gpu==2.0.1.post90 -i https://mirror.baidu.com/pypi/simple

# CUDA10.1
python -m pip install paddlepaddle-gpu==2.0.1.post101 -f https://mirror.baidu.com/pypi/simple

# CPU
python -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
```

更多安装方式例如conda或源码编译安装方法，请参考PaddlePaddle[安装文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/index_cn.html)

请确保您的PaddlePaddle安装成功并且版本不低于需求版本。使用以下命令进行验证。

```
# 在您的Python解释器中确认PaddlePaddle安装成功
>>> import paddle
>>> paddle.utils.run_check()

# 确认PaddlePaddle版本
python -c "import paddle; print(paddle.__version__)"
```
**注意**
1. 如果您希望在多卡环境下使用PaddleDetection，请首先安装NCCL

### 3. 安装PaddleDetection

```
# 克隆PaddleDetection仓库
cd <path/to/clone/PaddleDetection>
git clone https://github.com/PaddlePaddle/PaddleDetection.git

# 安装其他依赖
pip install -r requirements.txt

# 安装PaddleDetection
cd PaddleDetection
python setup.py install
```

**注意**

1. 由于原版cocoapi不支持windows，采用第三方实现版本，该版本仅支持Python3

    ```pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI```


安装后确认测试通过：

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

## 快速体验

**恭喜！**您已经成功安装了PaddleDetection，接下来快速体验目标检测效果

```
# 在GPU上预测一张图片
export CUDA_VISIBLE_DEVICES=0
python tools/infer.py -c configs/ppyolo/ppyolo.yml -o use_gpu=true weights=https://paddlemodels.bj.bcebos.com/object_detection/ppyolo.pdparams --infer_img=demo/000000014439.jpg
```

会在`output`文件夹下生成一个画有预测结果的同名图像。

结果如下图：

![](../images/000000014439.jpg)
