[English](README.md) | 简体中文
# PaddleDetection 服务化部署示例

本文档以PP-YOLOE模型(ppyoloe_crn_l_300e_coco)为例，进行详细介绍。其他PaddleDetection模型都已支持服务化部署，只需将下述命令中的模型和配置名字修改成要部署模型的名字。

PaddleDetection模型导出和预训练模型下载请看[PaddleDetection模型部署](../README.md)文档。

## 1. 部署环境准备
在服务化部署前，需确认

- 1. 服务化镜像的软硬件环境要求和镜像拉取命令请参考[FastDeploy服务化部署](https://github.com/PaddlePaddle/FastDeploy/blob/develop/serving/README_CN.md)


## 2. 启动服务

```bash
#下载部署示例代码
git clone https://github.com/PaddlePaddle/PaddleDetection.git
cd PaddleDetection/deploy/fastdeploy/serving

#下载PPYOLOE模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco.tgz
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
tar xvf ppyoloe_crn_l_300e_coco.tgz

# 将配置文件放入预处理目录
mv ppyoloe_crn_l_300e_coco/infer_cfg.yml models/preprocess/1/

# 将模型放入 models/runtime/1目录下, 并重命名为model.pdmodel和model.pdiparams
mv ppyoloe_crn_l_300e_coco/model.pdmodel models/runtime/1/model.pdmodel
mv ppyoloe_crn_l_300e_coco/model.pdiparams models/runtime/1/model.pdiparams

# 将ppdet和runtime中的ppyoloe配置文件重命名成标准的config名字
# 其他模型比如faster_rcc就将faster_rcnn_config.pbtxt重命名为config.pbtxt
cp models/ppdet/ppyoloe_config.pbtxt models/ppdet/config.pbtxt
cp models/runtime/ppyoloe_runtime_config.pbtxt models/runtime/config.pbtxt

# 注意： 由于mask_rcnn模型多一个输出，需要将后处理目录(models/postprocess)中的mask_config.pbtxt重命名为config.pbtxt

# 拉取fastdeploy镜像(x.y.z为镜像版本号，需替换成fastdeploy版本数字)
# GPU镜像
docker pull registry.baidubce.com/paddlepaddle/fastdeploy:x.y.z-gpu-cuda11.4-trt8.4-21.10
# CPU镜像
docker pull paddlepaddle/fastdeploy:z.y.z-cpu-only-21.10

# 运行容器.容器名字为 fd_serving, 并挂载当前目录为容器的 /serving 目录
nvidia-docker run -it --net=host --name fd_serving --shm-size="1g"  -v `pwd`/:/serving registry.baidubce.com/paddlepaddle/fastdeploy:x.y.z-gpu-cuda11.4-trt8.4-21.10  bash

# 启动服务(不设置CUDA_VISIBLE_DEVICES环境变量，会拥有所有GPU卡的调度权限)
CUDA_VISIBLE_DEVICES=0 fastdeployserver --model-repository=/serving/models
```
>> **注意**:

>> 由于mask_rcnn模型多一个输出，部署mask_rcnn需要将后处理目录(models/postprocess)中的mask_config.pbtxt重命名为config.pbtxt

>> 拉取镜像请看[服务化部署主文档](https://github.com/PaddlePaddle/FastDeploy/blob/develop/serving/README_CN.md)

>> 执行fastdeployserver启动服务出现"Address already in use", 请使用`--grpc-port`指定grpc端口号来启动服务，同时更改客户端示例中的请求端口号.

>> 其他启动参数可以使用 fastdeployserver --help 查看

服务启动成功后， 会有以下输出:
```
......
I0928 04:51:15.784517 206 grpc_server.cc:4117] Started GRPCInferenceService at 0.0.0.0:8001
I0928 04:51:15.785177 206 http_server.cc:2815] Started HTTPService at 0.0.0.0:8000
I0928 04:51:15.826578 206 http_server.cc:167] Started Metrics Service at 0.0.0.0:8002
```


## 3. 客户端请求

在物理机器中执行以下命令，发送grpc请求并输出结果
```
#下载测试图片
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg

#安装客户端依赖
python3 -m pip install tritonclient[all]

# 发送请求
python3 paddledet_grpc_client.py
```

发送请求成功后，会返回json格式的检测结果并打印输出:
```
output_name: DET_RESULT
[[159.93016052246094, 82.35527038574219, 199.8546600341797, 164.68682861328125],
... ...,
[60.200584411621094, 123.73260498046875, 108.83859252929688, 169.07467651367188]]
```

## 4. 配置修改

当前默认配置在GPU上运行Paddle引擎， 如果要在CPU或其他推理引擎上运行。 需要修改`models/runtime/config.pbtxt`中配置，详情请参考[配置文档](https://github.com/PaddlePaddle/FastDeploy/blob/develop/serving/docs/zh_CN/model_configuration.md)


## 5. 使用VisualDL进行可视化部署

可以使用VisualDL进行[Serving可视化部署](https://github.com/PaddlePaddle/FastDeploy/blob/develop/serving/docs/zh_CN/vdl_management.md)，上述启动服务、配置修改以及客户端请求的操作都可以基于VisualDL进行。

通过VisualDL的可视化界面对PaddleDetection进行服务化部署只需要如下三步：
```text
1. 载入模型库：./vision/detection/paddledetection/serving/models
2. 下载模型资源文件：点击preprocess模型，点击版本号1添加预训练模型，选择检测模型ppyoloe_crn_l_300e_coco进行下载，此时preprocess中将会有资源文件infer_cfg.yml。点击runtime模型，点击版本号1添加预训练模型，选择检测模型ppyoloe_crn_l_300e_coco进行下载，此时runtime中将会有资源文件model.pdmodel和model.pdiparams。
3. 设置启动配置文件：点击ensemble配置按钮，选择配置文件ppyoloe_config.pbtxt，并设为启动配置文件。点击runtime模型，选择配置文件ppyoloe_runtime_config.pbtxt，并设为启动配置文件。
4. 启动服务：点击启动服务按钮，输入启动参数。
```
 <p align="center">
  <img src="https://user-images.githubusercontent.com/22424850/211710983-2d1f1427-6738-409d-903b-2b4e4ab6cbfc.gif" width="100%"/>
</p>
