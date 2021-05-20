# PaddleDetection 预测部署

目前支持的部署方式有：
- `Paddle Inference预测库`部署:
  - `Python`语言部署，支持`CPU`、`GPU`和`XPU`环境，参考文档[python部署](python/README.md)。
  - `C++`语言部署 ，支持`CPU`、`GPU`和`XPU`环境，支持在`Linux`、`Windows`系统下部署，支持`NV Jetson`嵌入式设备上部署。请参考文档[C++部署](cpp/README.md)。
  - `TensorRT`加速：请参考文档[TensorRT预测部署教程](TENSOR_RT.md)
- 服务器端部署：使用[PaddleServing](./serving/README.md)部署。
- 手机移动端部署：使用[Paddle-Lite](./lite/README.md) 在手机移动端部署。


## 1.模型导出

使用`tools/export_model.py`脚本导出模型已经部署时使用的配置文件，配置文件名字为`infer_cfg.yml`。模型导出脚本如下：
```bash
# 导出YOLOv3模型
python tools/export_model.py -c configs/yolov3/yolov3_darknet53_270e_coco.yml -o weights=weights/yolov3_darknet53_270e_coco.pdparams
```
预测模型会导出到`output_inference/yolov3_darknet53_270e_coco`目录下，分别为`infer_cfg.yml`, `model.pdiparams`,  `model.pdiparams.info`, `model.pdmodel`。

如果需要导出`PaddleServing`格式的模型，需要设置`export_serving_model=True`:
```buildoutcfg
python tools/export_model.py -c configs/yolov3/yolov3_darknet53_270e_coco.yml -o weights=weights/yolov3_darknet53_270e_coco.pdparams --export_serving_model=True
```
预测模型会导出到`output_inference/yolov3_darknet53_270e_coco`目录下，分别为`infer_cfg.yml`, `model.pdiparams`,  `model.pdiparams.info`, `model.pdmodel`, `serving_client/`文件夹, `serving_server/`文件夹。

模型导出具体请参考文档[PaddleDetection模型导出教程](EXPORT_MODEL.md)。

## 2.部署环境准备

- Python预测：在python环境下安装PaddlePaddle环境即可，如需TensorRT预测，在[Paddle Release版本](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/Tables.html#whl-release)中下载合适的wheel包即可。

- C++预测库：请从[这里](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/05_inference_deployment/inference/build_and_install_lib_cn.html)，如果需要使用TensorRT，请下载带有TensorRT编译的预测库。您也可以自行编译，编译过程请参考[Paddle源码编译](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/compile/linux-compile.html)。
**注意:**  Paddle预测库版本需要>=2.0

- PaddleServing部署
  请选择PaddleServing>0.5.0以上版本，具体可参考[PaddleServing安装文档](https://github.com/PaddlePaddle/Serving/blob/develop/README.md#installation)。

- Paddle-Lite部署
  Paddle-Lite支持OP列表请参考：[Paddle-Lite支持的OP列表](https://paddle-lite.readthedocs.io/zh/latest/source_compile/library.html) ，请跟进所部署模型中使用到的op选择Paddle-Lite版本。

- NV Jetson部署
  Paddle官网提供在NV Jetson平台上已经编译好的预测库，[Paddle NV Jetson预测库](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/05_inference_deployment/inference/build_and_install_lib_cn.html)。若列表中没有您需要的预测库，您可以在您的平台上自行编译，编译过程请参考[Paddle源码编译](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/compile/linux-compile.html)。

## 3.部署预测
- Python部署：使用`deploy/python/infer.py`进行预测，可具体参考[python部署文档](python/README.md)。
```shell
python deploy/python/infer.py --model_dir=/path/to/models --image_file=/path/to/image --use_gpu=(False/True)
```

- C++部署，先使用跨平台编译工具`CMake`根据`CMakeLists.txt`生成`Makefile`，支持[Windows](cpp/docs/windows_vs2019_build.md)、[Linux](cpp/docs/linux_build.md)、[NV Jetson](cpp/docs/Jetson_build.md)平台部署，然后进行编译产出可执行文件。可以直接使用`cpp/scripts/build.sh`脚本编译：
```buildoutcfg
cd cpp
sh scripts/build.sh
```

- PaddleServing部署请参考，[PaddleServing部署](./serving/README.md)部署。

- 手机移动端部署，请参考[Paddle-Lite-Demo](https://github.com/PaddlePaddle/Paddle-Lite-Demo)部署。

## 4.Benchmark测试
- 使用导出的模型，运行Benchmark批量测试脚本：
```shell
sh deploy/benchmark/benchmark.sh {model_dir} {model_name}
```
**注意** 如果是量化模型，请使用`deploy/benchmark/benchmark_quant.sh`脚本。
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
