[English](README.md) | 简体中文

# PaddleDetection RKNPU2部署示例

## 1. 说明  
RKNPU2 提供了一个高性能接口来访问 Rockchip NPU，支持如下硬件的部署
- RK3566/RK3568
- RK3588/RK3588S
- RV1103/RV1106

在RKNPU2上已经通过测试的PaddleDetection模型如下:

- Picodet
- PPYOLOE(int8)
- YOLOV8

如果你需要查看详细的速度信息，请查看[RKNPU2模型速度一览表](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/rknpu2/rknpu2.md)

## 2. 使用预导出的模型列表

### ONNX模型转RKNN模型

为了方便大家使用，我们提供了python脚本，通过我们预配置的config文件，你将能够快速地转换ONNX模型到RKNN模型

```bash
python tools/rknpu2/export.py --config_path tools/rknpu2/config/picodet_s_416_coco_lcnet_unquantized.yaml \
                              --target_platform rk3588
```

### RKNN模型列表

为了方便大家测试，我们提供picodet和ppyoloe两个模型，解压后即可使用:

| 模型名称                        | 下载地址                                                                              |
|-----------------------------|-----------------------------------------------------------------------------------|
| picodet_s_416_coco_lcnet    | https://bj.bcebos.com/paddlehub/fastdeploy/rknpu2/picodet_s_416_coco_lcnet.zip    |
| ppyoloe_plus_crn_s_80e_coco | https://bj.bcebos.com/paddlehub/fastdeploy/rknpu2/ppyoloe_plus_crn_s_80e_coco.zip |


## 3. 自行导出PaddleDetection部署模型以及转换模型

RKNPU部署模型前需要将Paddle模型转换成RKNN模型，具体步骤如下:

* Paddle动态图模型转换为ONNX模型，请参考[PaddleDetection导出模型](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/deploy/EXPORT_MODEL.md)
,注意在转换时请设置**export.nms=True**.
* ONNX模型转换RKNN模型的过程，请参考[转换文档](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/rknpu2/export.md)进行转换。

### 3.1 模型转换example

#### 3.1.1 注意点

PPDetection模型在RKNPU2上部署时要注意以下几点:

* 模型导出需要包含Decode
* 由于RKNPU2不支持NMS，因此输出节点必须裁剪至NMS之前
* 由于RKNPU2 Div算子的限制，模型的输出节点需要裁剪至Div算子之前

#### 3.1.2 Paddle模型转换为ONNX模型

由于Rockchip提供的rknn-toolkit2工具暂时不支持Paddle模型直接导出为RKNN模型，因此需要先将Paddle模型导出为ONNX模型，再将ONNX模型转为RKNN模型。

```bash
# 以Picodet为例
# 下载Paddle静态图模型并解压
wget https://paddledet.bj.bcebos.com/deploy/Inference/picodet_s_416_coco_lcnet.tar
tar xvf picodet_s_416_coco_lcnet.tar

# 静态图转ONNX模型，注意，这里的save_file请和压缩包名对齐
paddle2onnx --model_dir picodet_s_416_coco_lcnet \
            --model_filename model.pdmodel \
            --params_filename model.pdiparams \
            --save_file picodet_s_416_coco_lcnet/picodet_s_416_coco_lcnet.onnx \
            --enable_dev_version True

# 固定shape
python -m paddle2onnx.optimize --input_model picodet_s_416_coco_lcnet/picodet_s_416_coco_lcnet.onnx \
                                --output_model picodet_s_416_coco_lcnet/picodet_s_416_coco_lcnet.onnx \
                                --input_shape_dict "{'image':[1,3,416,416], 'scale_factor':[1,2]}"
```

#### 3.1.3 编写yaml文件

**修改normalize参数**

如果你需要在NPU上执行normalize操作，请根据你的模型配置normalize参数，例如:

```yaml
mean:
  -
    - 123.675
    - 116.28
    - 103.53
std:
  -
    - 58.395
    - 57.12
    - 57.375
```

**修改outputs参数**

由于Paddle2ONNX版本的不同，转换模型的输出节点名称也有所不同，请使用[Netron](https://netron.app)对模型进行可视化，并找到以下蓝色方框标记的NonMaxSuppression节点，红色方框的节点名称即为目标名称。

## 4. 模型可视化
例如，使用Netron可视化后，得到以下图片:

![](https://ai-studio-static-online.cdn.bcebos.com/8bce6b904a6b479e8b30da9f7c719fad57517ffb2f234aeca3b8ace0761754d5)

找到蓝色方框标记的NonMaxSuppression节点，可以看到红色方框标记的两个节点名称为p2o.Div.79和p2o.Concat.9,因此需要修改outputs参数，修改后如下:

```yaml
outputs_nodes:
  - 'p2o.Mul.179'
  - 'p2o.Concat.9'
```


## 5. 详细的部署示例  
- [RKNN总体部署教程](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/rknpu2/rknpu2.md)
- [C++部署](cpp)
- [Python部署](python)
