# PicoDet OpenVINO Benchmark Demo

本文件夹提供利用[Intel's OpenVINO Toolkit](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html)进行PicoDet测速的Benchmark Demo与带后处理的模型Inference Demo。

## 安装 OpenVINO Toolkit

前往 [OpenVINO HomePage](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html)，下载对应版本并安装。

本demo安装的是 OpenVINO 2022.1.0，可直接运行如下指令安装：
```shell
pip install openvino==2022.1.0
```

详细安装步骤，可参考[OpenVINO官网](https://docs.openvinotoolkit.org/latest/get_started_guides.html)

## Benchmark测试

- 准备测试模型：根据[PicoDet](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/picodet)中【导出及转换模型】步骤，采用不包含后处理的方式导出模型（`-o export.benchmark=True` ），并生成待测试模型简化后的onnx模型（可在下文链接中直接下载）。同时在本目录下新建```out_onnxsim```文件夹，将导出的onnx模型放在该目录下。

- 准备测试所用图片：本demo默认利用PaddleDetection/demo/[000000014439.jpg](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.6/demo/000000014439.jpg)

- 在本目录下直接运行：

```shell
# Linux
python openvino_benchmark.py --img_path ../../../../demo/000000014439.jpg --onnx_path out_onnxsim/picodet_s_320_coco_lcnet.onnx --in_shape 320
# Windows
python openvino_benchmark.py --img_path ..\..\..\..\demo\000000014439.jpg --onnx_path out_onnxsim\picodet_s_320_coco_lcnet.onnx --in_shape 320
```
- 注意：```--in_shape```为对应模型输入size，默认为320

## 真实图片测试(网络包含后处理，但不包含NMS)

- 准备测试模型：根据[PicoDet](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/picodet)中【导出及转换模型】步骤，采用**包含后处理**但**不包含NMS**的方式导出模型（`-o export.benchmark=False export.nms=False` ），并生成待测试模型简化后的onnx模型（可在下文链接中直接下载）。同时在本目录下新建```out_onnxsim_infer```文件夹，将导出的onnx模型放在该目录下。

- 准备测试所用图片：默认利用../../demo_onnxruntime/imgs/[bus.jpg](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.6/deploy/third_engine/demo_onnxruntime/imgs/bus.jpg)

```shell
# Linux
python openvino_infer.py --img_path ../../demo_onnxruntime/imgs/bus.jpg --onnx_path out_onnxsim_infer/picodet_s_320_postproccesed_woNMS.onnx --in_shape 320
# Windows
python openvino_infer.py --img_path ..\..\demo_onnxruntime\imgs\bus.jpg --onnx_path out_onnxsim_infer\picodet_s_320_postproccesed_woNMS.onnx --in_shape 320
```

### 真实图片测试(网络不包含后处理)

```shell
# Linux
python openvino_benchmark.py --benchmark 0 --img_path ../../../../demo/000000014439.jpg --onnx_path out_onnxsim/picodet_s_320_coco_lcnet.onnx --in_shape 320
# Windows
python openvino_benchmark.py --benchmark 0 --img_path ..\..\..\..\demo\000000014439.jpg --onnx_path out_onnxsim\picodet_s_320_coco_lcnet.onnx --in_shape 320
```

- 结果：
    <div align="center">
      <img src="../../../../docs/images/res.jpg" height="500px" >
    </div>

## Benchmark结果

- 测速结果如下：

| 模型     | 输入尺寸 | ONNX  | 预测时延<sup><small>[CPU](#latency)|
| :-------- | :--------: | :---------------------: | :----------------: |
| PicoDet-XS |  320*320   | [( w/ 后处理;w/o NMS)](https://paddledet.bj.bcebos.com/deploy/third_engine/picodet_xs_320_lcnet_postproccesed_woNMS.onnx) &#124; [( w/o 后处理)](https://paddledet.bj.bcebos.com/deploy/third_engine/picodet_xs_320_coco_lcnet.onnx) | 3.9ms |
| PicoDet-XS |  416*416   | [( w/ 后处理;w/o NMS)](https://paddledet.bj.bcebos.com/deploy/third_engine/picodet_xs_416_lcnet_postproccesed_woNMS.onnx) &#124; [( w/o 后处理)](https://paddledet.bj.bcebos.com/deploy/third_engine/picodet_xs_416_coco_lcnet.onnx) | 6.1ms |
| PicoDet-S |  320*320   | [( w/ 后处理;w/o NMS)](https://paddledet.bj.bcebos.com/deploy/third_engine/picodet_s_320_lcnet_postproccesed_woNMS.onnx) &#124; [( w/o 后处理)](https://paddledet.bj.bcebos.com/deploy/third_engine/picodet_s_320_coco_lcnet.onnx) |     4.8ms |
| PicoDet-S |  416*416   |  [( w/ 后处理;w/o NMS)](https://paddledet.bj.bcebos.com/deploy/third_engine/picodet_s_416_lcnet_postproccesed_woNMS.onnx) &#124; [( w/o 后处理)](https://paddledet.bj.bcebos.com/deploy/third_engine/picodet_s_416_coco_lcnet.onnx) |     6.6ms |
| PicoDet-M |  320*320   | [( w/ 后处理;w/o NMS)](https://paddledet.bj.bcebos.com/deploy/third_engine/picodet_m_320_lcnet_postproccesed_woNMS.onnx) &#124; [( w/o 后处理)](https://paddledet.bj.bcebos.com/deploy/third_engine/picodet_m_320_coco_lcnet.onnx) | 8.2ms  |
| PicoDet-M |  416*416   | [( w/ 后处理;w/o NMS)](https://paddledet.bj.bcebos.com/deploy/third_engine/picodet_m_416_lcnet_postproccesed_woNMS.onnx) &#124; [( w/o 后处理)](https://paddledet.bj.bcebos.com/deploy/third_engine/picodet_m_416_coco_lcnet.onnx) | 12.7ms |
| PicoDet-L |  320*320   | [( w/ 后处理;w/o NMS)](https://paddledet.bj.bcebos.com/deploy/third_engine/picodet_l_320_lcnet_postproccesed_woNMS.onnx) &#124; [( w/o 后处理)](https://paddledet.bj.bcebos.com/deploy/third_engine/picodet_l_320_coco_lcnet.onnx) | 11.5ms |
| PicoDet-L |  416*416   | [( w/ 后处理;w/o NMS)](https://paddledet.bj.bcebos.com/deploy/third_engine/picodet_l_416_lcnet_postproccesed_woNMS.onnx) &#124; [( w/o 后处理)](https://paddledet.bj.bcebos.com/deploy/third_engine/picodet_l_416_coco_lcnet.onnx) |     20.7ms |
| PicoDet-L |  640*640   | [( w/ 后处理;w/o NMS)](https://paddledet.bj.bcebos.com/deploy/third_engine/picodet_l_640_lcnet_postproccesed_woNMS.onnx) &#124; [( w/o 后处理)](https://paddledet.bj.bcebos.com/deploy/third_engine/picodet_l_640_coco_lcnet.onnx) |     62.5ms |

- <a name="latency">测试环境：</a> 英特尔酷睿i7 10750H CPU。
