# PaddleDetection 预测部署
训练得到一个满足要求的模型后，如果想要将该模型部署到已选择的平台上，需要通过`tools/export_model.py`将模型导出预测部署的模型和配置文件。
并在同一文件夹下导出预测时使用的配置文件，配置文件名为`infer_cfg.yml`。

## `PaddleDetection`目前支持的部署方式按照部署设备可以分为：
- 在本机`python`语言部署，支持在有`python paddle`(支持`CPU`、`GPU`)环境下部署，有两种方式：
    - 使用`tools/infer.py`，此种方式依赖`PaddleDetection`代码库。
    - 将模型导出，使用`deploy/python/infer.py`，此种方式不依赖`PaddleDetection`代码库，可以单个`python`文件部署。
- 在本机`C++`语言使用`paddle inference`预测库部署，支持在`Linux`和`Windows`系统下部署。请参考文档[C++部署](cpp/README.md)。
- 在服务器端以服务形式部署，使用[PaddleServing](./serving/README.md)部署。
- 在手机移动端部署，使用[Paddle-Lite](https://github.com/PaddlePaddle/Paddle-Lite) 在手机移动端部署。
  常见模型部署Demo请参考[Paddle-Lite-Demo](https://github.com/PaddlePaddle/Paddle-Lite-Demo) 。
- `NV Jetson`嵌入式设备上部署

## 模型导出
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

## 如何选择部署时依赖库的版本

### CUDA、cuDNN、TensorRT版本选择
由于CUDA、cuDNN、TENSORRT不一定都是向前兼容的，需要使用与编译Paddle预测库使用的环境完全一致的环境进行部署。

### 部署时预测库版本、预测引擎版本选择

- Linux、Windows平台下C++部署，需要使用Paddle预测库进行部署。
  （1）Paddle官网提供在不同平台、不同环境下编译好的预测库，您可以直接使用，请在这里[Paddle预测库](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/05_inference_deployment/inference/build_and_install_lib_cn.html) 选择。
  （2）如果您将要部署的平台环境，Paddle官网上没有提供已编译好的预测库，您可以自行编译，编译过程请参考[Paddle源码编译](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/compile/linux-compile.html)。

**注意:**  Paddle预测库版本需要>=2.0

- Python语言部署，需要在对应平台上安装Paddle Python包。如果Paddle官网上没有提供该平台下的Paddle Python包，您可以自行编译，编译过程请参考[Paddle源码编译](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/compile/linux-compile.html)。

- PaddleServing部署
  PaddleServing 0.4.0是基于Paddle 1.8.4开发，PaddleServing 0.5.0是基于Paddle2.0开发。

- Paddle-Lite部署
  Paddle-Lite支持OP列表请参考：[Paddle-Lite支持的OP列表](https://paddle-lite.readthedocs.io/zh/latest/source_compile/library.html) ，请跟进所部署模型中使用到的op选择Paddle-Lite版本。

- NV Jetson部署
  Paddle官网提供在NV Jetson平台上已经编译好的预测库，[Paddle NV Jetson预测库](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/05_inference_deployment/inference/build_and_install_lib_cn.html) 。
  若列表中没有您需要的预测库，您可以在您的平台上自行编译，编译过程请参考[Paddle源码编译](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/compile/linux-compile.html)。


## 部署
- C++部署，先使用跨平台编译工具`CMake`根据`CMakeLists.txt`生成`Makefile`，支持`Windows、Linux、NV Jetson`平台，然后进行编译产出可执行文件。可以直接使用`cpp/scripts/build.sh`脚本编译：
```buildoutcfg
cd cpp
sh scripts/build.sh
```

- Python部署，可以使用使用`tools/infer.py`（以来PaddleDetection源码）部署，或者使用`deploy/python/infer.py`单文件部署

- PaddleServing部署请参考，[PaddleServing部署](./serving/README.md)部署。

- 手机移动端部署，请参考[Paddle-Lite-Demo](https://github.com/PaddlePaddle/Paddle-Lite-Demo)部署。


## 常见问题QA
- 1、`Paddle 1.8.4`训练的模型，可以用`Paddle2.0`部署吗？
  Paddle 2.0是兼容Paddle 1.8.4的，因此是可以的。但是部分模型(如SOLOv2)使用到了Paddle 2.0中新增OP，这类模型不可以。

- 2、Windows编译时，预测库是VS2015编译的，选择VS2017或VS2019会有问题吗？
  关于VS兼容性问题请参考：[C++Visual Studio 2015、2017和2019之间的二进制兼容性](https://docs.microsoft.com/zh-cn/cpp/porting/binary-compat-2015-2017?view=msvc-160)

- 3、cuDNN 8.0.4连续预测会发生内存泄漏吗？
  经QA测试，发现cuDNN 8系列连续预测时都有内存泄漏问题，且cuDNN 8性能差于cuDNN 7，推荐使用CUDA + cuDNN7.6.4的方式进行部署。
