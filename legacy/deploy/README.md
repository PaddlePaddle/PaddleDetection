# PaddleDetection 预测部署

`PaddleDetection`目前支持：
- 使用`Python`和`C++`部署在`Windows` 和`Linux` 上运行
- [在线服务化部署](./serving/README.md)
- [移动端部署](https://github.com/PaddlePaddle/Paddle-Lite-Demo)

## 模型导出
训练得到一个满足要求的模型后，如果想要将该模型接入到C++服务器端预测库或移动端预测库，需要通过`tools/export_model.py`导出该模型。

- [导出教程](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/static/docs/advanced_tutorials/deploy/EXPORT_MODEL.md)

模型导出后, 目录结构如下(以`yolov3_darknet`为例):
```
yolov3_darknet # 模型目录
├── infer_cfg.yml # 模型配置信息
├── __model__     # 模型文件
└── __params__    # 参数文件
```

预测时，该目录所在的路径会作为程序的输入参数。

## 预测部署
- [1. Python预测(支持 Linux 和 Windows)](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/static/deploy/python)
- [2. C++预测(支持 Linux 和 Windows)](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/static/deploy/cpp)
- [3. 在线服务化部署](./serving/README.md)
- [4. 移动端部署](https://github.com/PaddlePaddle/Paddle-Lite-Demo)
- [5. Jetson设备部署](./cpp/docs/Jetson_build.md)
