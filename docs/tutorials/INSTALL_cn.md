[English](INSTALL.md) | 简体中文


# 安装文档



## 环境要求

- PaddlePaddle 2.3.2
- OS 64位操作系统
- Python 3(3.5.1+/3.6/3.7/3.8/3.9/3.10)，64位版本
- pip/pip3(9.0.1+)，64位版本
- CUDA >= 10.2
- cuDNN >= 7.6

PaddleDetection 依赖 PaddlePaddle 版本关系：

|  PaddleDetection版本  | PaddlePaddle版本  |    备注    |
| :------------------: | :---------------: | :-------: |
|    develop           |       >=2.3.2     |     默认使用动态图模式    |
|    release/2.6       |       >=2.3.2     |     默认使用动态图模式    |
|    release/2.5       |       >= 2.2.2    |     默认使用动态图模式    |
|    release/2.4       |       >= 2.2.2    |     默认使用动态图模式    |
|    release/2.3       |       >= 2.2.0rc  |     默认使用动态图模式    |
|    release/2.2       |       >= 2.1.2    |     默认使用动态图模式    |
|    release/2.1       |       >= 2.1.0    |     默认使用动态图模式    |
|    release/2.0       |       >= 2.0.1    |     默认使用动态图模式    |
|    release/2.0-rc    |       >= 2.0.1    |     --    |
|    release/0.5       |       >= 1.8.4    |  大部分模型>=1.8.4即可运行，Cascade R-CNN系列模型与SOLOv2依赖2.0.0.rc版本 |
|    release/0.4       |       >= 1.8.4    |  PP-YOLO依赖1.8.4 |
|    release/0.3       |        >=1.7      |     --    |

## 安装说明

### 1. 安装PaddlePaddle

```
# CUDA10.2
python -m pip install paddlepaddle-gpu==2.3.2 -i https://mirror.baidu.com/pypi/simple

# CPU
python -m pip install paddlepaddle==2.3.2 -i https://mirror.baidu.com/pypi/simple
```
- 更多CUDA版本或环境快速安装，请参考[PaddlePaddle快速安装文档](https://www.paddlepaddle.org.cn/install/quick)
- 更多安装方式例如conda或源码编译安装方法，请参考[PaddlePaddle安装文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/index_cn.html)

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

### 2. 安装PaddleDetection




**注意：** pip安装方式只支持Python3



```
# 克隆PaddleDetection仓库
cd <path/to/clone/PaddleDetection>
git clone https://github.com/PaddlePaddle/PaddleDetection.git

# 安装其他依赖
cd PaddleDetection
pip install -r requirements.txt

# 编译安装paddledet
python setup.py install
```

**注意**
1. 如果github下载代码较慢，可尝试使用[gitee](https://gitee.com/PaddlePaddle/PaddleDetection.git)或者[代理加速](https://doc.fastgit.org/zh-cn/guide.html)。

1. 若您使用的是Windows系统，由于原版cocoapi不支持Windows，`pycocotools`依赖可能安装失败，可采用第三方实现版本，该版本仅支持Python3

    ```pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI```

2. 若您使用的是Python <= 3.6的版本，安装`pycocotools`可能会报错`distutils.errors.DistutilsError: Could not find suitable distribution for Requirement.parse('cython>=0.27.3')`, 您可通过先安装`cython`如`pip install cython`解决该问题


安装后确认测试通过：

```
python ppdet/modeling/tests/test_architectures.py
```

测试通过后会提示如下信息：

```
.......
----------------------------------------------------------------------
Ran 7 tests in 12.816s
OK
```

## 使用Docker镜像
> 如果您没有Docker运行环境，请参考[Docker官网](https://www.docker.com/)进行安装。

我们提供了包含最新 PaddleDetection 代码的docker镜像，并预先安装好了所有的环境和库依赖，您只需要**拉取docker镜像**，然后**运行docker镜像**，无需其他任何额外操作，即可开始使用PaddleDetection的所有功能。

在[Docker Hub](https://hub.docker.com/repository/docker/paddlecloud/paddledetection)中获取这些镜像及相应的使用指南，包括CPU、GPU、ROCm版本。
如果您对自动化制作docker镜像感兴趣，或有自定义需求，请访问[PaddlePaddle/PaddleCloud](https://github.com/PaddlePaddle/PaddleCloud/tree/main/tekton)做进一步了解。

## 快速体验

**恭喜！** 您已经成功安装了PaddleDetection，接下来快速体验目标检测效果

```
# 在GPU上预测一张图片
export CUDA_VISIBLE_DEVICES=0
python tools/infer.py -c configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml -o use_gpu=true weights=https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_1x_coco.pdparams --infer_img=demo/000000014439.jpg
```

会在`output`文件夹下生成一个画有预测结果的同名图像。

结果如下图：

![](../images/000000014439.jpg)
