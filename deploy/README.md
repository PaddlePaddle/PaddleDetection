# PaddleDetection 预测部署

PaddleDetection提供了PaddleInference、PaddleServing、PaddleLite多种部署形式，支持服务端、移动端、嵌入式等多种平台，提供了完善的Python和C++部署方案。

## PaddleDetection支持的部署形式说明
|形式|语言|教程|设备/平台|
|-|-|-|-|
|PaddleInference|Python|已完善|linux(ARM\X86)、windows
|PaddleInference|C++|已完善|linux(ARM\X86)、windows|
|PaddleServing|Python|已完善|linux(ARM\X86)、windows|
|PaddleLite|Python|已完善|Android、IOS、FPGA、RK...


## 1.Paddle Inference部署

### 1.1 导出PaddleInference的模型

使用`tools/export_model.py`脚本导出模型已经部署时使用的配置文件，配置文件名字为`infer_cfg.yml`。模型导出脚本如下：
```bash
# 导出YOLOv3模型
python tools/export_model.py -c configs/yolov3/yolov3_darknet53_270e_coco.yml -o weights=weights/yolov3_darknet53_270e_coco.pdparams
```
预测模型会导出到`inference/yolov3_darknet53_270e_coco`目录下，分别为`infer_cfg.yml`, `model.pdiparams`,  `model.pdiparams.info`, `model.pdmodel`。
模型导出具体请参考文档[PaddleDetection模型导出教程](EXPORT_MODEL.md)。

### 1.2 使用PaddleInference进行预测
* Python部署 支持`CPU`、`GPU`和`XPU`环境，支持，windows、linux系统，支持NV Jetson嵌入式设备上部署。参考文档[python部署](python/README.md)
* C++部署 支持`CPU`、`GPU`和`XPU`环境，支持，windows、linux系统，支持NV Jetson嵌入式设备上部署。参考文档[C++部署](cpp/README.md)
* PaddleDetection支持TensorRT加速,相关文档请参考[TensorRT预测部署教程](TENSOR_RT.md)

##  2.PaddleServing部署
### 2.1 导出PaddleInference的模型

如果需要导出`PaddleServing`格式的模型，需要设置`export_serving_model=True`:
```buildoutcfg
python tools/export_model.py -c configs/yolov3/yolov3_darknet53_270e_coco.yml -o weights=weights/yolov3_darknet53_270e_coco.pdparams --export_serving_model=True
```
预测模型会导出到`inference/yolov3_darknet53_270e_coco`目录下，分别为`infer_cfg.yml`, `model.pdiparams`,  `model.pdiparams.info`, `model.pdmodel`, `serving_client/`文件夹, `serving_server/`文件夹。

模型导出具体请参考文档[PaddleDetection模型导出教程](EXPORT_MODEL.md)。

### 2.2 使用PaddleServing进行预测
* [安装PaddleServing](https://github.com/PaddlePaddle/Serving/blob/develop/README.md#installation)
* [使用PaddleServing](./serving/README.md)


## 3.PaddleLite部署
* 手机移动端部署：使用[Paddle-Lite](https://github.com/PaddlePaddle/Paddle-Lite) 在手机移动端部署。

## 4.BenchMark测试
- 使用导出的模型，运行BenchMark批量测试脚本：
```shell
sh deploy/benchmark/benchmark_ppdet.sh {model_dir} {model_name}
```
**注意** 如果是量化模型，请使用`deploy/benchmark/benchmark_ppdet_quant.sh`脚本。
- 将测试结果log导出至Excel中：
```
python deploy/benchmark/log_parser_excel.py --log_path=./output_pipeline --output_name=benchmark_excel.xlsx
```

## 5.常见问题QA
- 1、`Paddle 1.8.4`训练的模型，可以用`Paddle2.0`部署吗？
  Paddle 2.0是兼容Paddle 1.8.4的，因此是可以的。但是部分模型(如SOLOv2)使用到了Paddle 2.0中新增OP，这类模型不可以。

- 2、Windows编译时，预测库是VS2015编译的，选择VS2017或VS2019会有问题吗？
  关于VS兼容性问题请参考：[C++Visual Studio 2015、2017和2019之间的二进制兼容性](https://docs.microsoft.com/zh-cn/cpp/porting/binary-compat-2015-2017?view=msvc-160)

- 3、cuDNN 8.0.4连续预测会发生内存泄漏吗？
  经QA测试，发现cuDNN 8系列连续预测时都有内存泄漏问题，且cuDNN 8性能差于cuDNN 7，推荐使用CUDA + cuDNN7.6.4的方式进行部署。
