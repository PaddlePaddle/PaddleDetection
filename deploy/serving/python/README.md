# Python Serving预测部署

## 1. 简介
Paddle Serving是飞桨开源的服务化部署框架，提供了C++ Serving和Python Pipeline两套框架，
C++ Serving框架更倾向于追求极致性能，Python Pipeline框架倾向于二次开发的便捷性。
旨在帮助深度学习开发者和企业提供高性能、灵活易用的工业级在线推理服务，助力人工智能落地应用。

更多关于Paddle Serving的介绍，可以参考[Paddle Serving官网repo](https://github.com/PaddlePaddle/Serving)。

本文档主要介绍利用Python Pipeline框架实现模型（以yolov3_darknet53_270e_coco为例）的服务化部署。

## 2. Python Serving预测部署

#### 2.1 Python 服务化部署样例程序介绍
服务化部署的样例程序的目录地址为：`deploy/serving/python`
```shell
deploy/
├── serving/
│   ├── python/                       # Python 服务化部署样例程序目录
│   │   ├──config.yml                 # 服务端模型预测相关配置文件
│   │   ├──pipeline_http_client.py    # 客户端代码
│   │   ├──postprocess_ops.py         # 用户自定义后处理代码
│   │   ├──preprocess_ops.py          # 用户自定义预处理代码
│   │   ├──README.md                  # 说明文档
│   │   ├──web_service.py             # 服务端代码
│   ├── cpp/                          # C++ 服务化部署样例程序目录
│   │   ├──preprocess/                # C++ 自定义OP
│   │   ├──build_server.sh            # C++ Serving 编译脚本
│   │   ├──serving_client.py          # 客户端代码
│   │   └── ...
│   └── ...
└── ...
```

### 2.2 环境准备
安装Paddle Serving四个安装包的最新版本，
分别是：paddle-serving-server(CPU/GPU版本二选一),
paddle-serving-client, paddle-serving-app和paddlepaddle(CPU/GPU版本二选一)。
```commandline
pip install paddle-serving-client
# pip install paddle-serving-server # CPU
pip install paddle-serving-server-gpu # GPU 默认 CUDA10.2 + TensorRT6，其他环境需手动指定版本号
pip install paddle-serving-app
# pip install paddlepaddle # CPU
pip install paddlepaddle-gpu
```
您可能需要使用国内镜像源（例如百度源, 在pip命令中添加`-i https://mirror.baidu.com/pypi/simple`）来加速下载。
Paddle Serving Server更多不同运行环境的whl包下载地址，请参考：[下载页面](https://github.com/PaddlePaddle/Serving/blob/v0.7.0/doc/Latest_Packages_CN.md)
PaddlePaddle更多版本请参考[官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)

### 2.3 服务化部署模型导出
导出步骤参考文档[PaddleDetection部署模型导出教程](../../EXPORT_MODEL.md),
导出服务化部署模型需要添加`--export_serving_model True`参数，导出示例如下:
```commandline
python tools/export_model.py -c configs/yolov3/yolov3_darknet53_270e_coco.yml \
                             --export_serving_model True \
                             -o weights=https://paddledet.bj.bcebos.com/models/yolov3_darknet53_270e_coco.pdparams
```

### 2.4 启动服务端模型预测服务
当完成以上环境准备和模型导出后，可以按如下命令启动模型预测服务：
```commandline
python deploy/serving/python/web_service.py --model_dir output_inference/yolov3_darknet53_270e_coco &
```
服务端模型预测相关配置可在[config.yml](./config.yml)中修改，
开发者只需要关注如下配置：http_port（服务的http端口），device_type（计算硬件类型），devices（计算硬件ID）。

### 2.5 启动客户端访问
当成功启动了模型预测服务，可以按如下命令启动客户端访问服务：
```commandline
python deploy/serving/python/pipeline_http_client.py --image_file demo/000000014439.jpg
```
