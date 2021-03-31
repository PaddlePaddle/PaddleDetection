# 人像圣诞特效自动生成工具
通过SOLOv2实例分割模型分割人像，并通过BlazeFace关键点模型检测人脸关键点，然后根据两个模型输出结果更换圣诞风格背景并为人脸加上圣诞老人胡子、圣诞眼镜及圣诞帽等特效。本项目通过PaddleHub可直接发布Server服务，供本地调试与前端直接调用接口。您可通过以下二维码中微信小程序直接体验：

<div align="center">
  <img src="demo_images/wechat_app.jpeg" width='400'/>
</div>

## 环境搭建

### 环境依赖

- paddlepaddle >= 2.0.0rc0

- paddlehub >= 2.0.0b1

### 模型准备
- 首先要获取模型，可在[模型配置文件](../../configs)里配置`solov2`与`blazeface_keypoint`，训练模型，并[导出模型](../../docs/advanced_tutorials/deploy/EXPORT_MODEL.md)。也可直接下载我们准备好模型：
[blazeface_keypoint模型](https://paddlemodels.bj.bcebos.com/object_detection/application/blazeface_keypoint.tar)和
[solov2模型](https://paddlemodels.bj.bcebos.com/object_detection/application/solov2_r101_vd_fpn_3x.tar)。
**注意：** 下载的模型需要解压后使用。

- 然后将两个模型文件夹中的文件(`infer_cfg.yml`、`__model__`和`__params__`)分别拷贝至`blazeface/blazeface_keypoint/` 和 `solov2/solov2_r101_vd_fpn_3x/`文件夹内。

### hub安装blazeface和solov2模型

```shell
hub install solov2
hub install blazeface
```

### hub安装solov2_blazeface圣诞特效自动生成串联模型

```shell
$ hub install solov2_blazeface
```
## 开始测试

### 本地测试

```shell
python test_main.py
```
运行成功后，预测结果会保存到`chrismas_final.png`。

### serving测试

- step1: 启动服务

```shell
export CUDA_VISIBLE_DEVICES=0
hub serving start -m solov2_blazeface -p 8880
```

- step2: 在服务端发送预测请求

```shell
python test_server.py
```
运行成功后，预测结果会保存到`chrismas_final.png`。

## 效果展示

<div align="center">
  <img src="demo_images/test.jpg" height="600px" ><img src="demo_images/result.png" height="600px" >
</div>
