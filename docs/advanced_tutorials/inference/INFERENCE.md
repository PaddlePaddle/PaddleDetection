# 模型预测

本篇教程使用Python API对[导出模型](EXPORT_MODEL.md)保存的inference_model进行预测。

在PaddlePaddle中预测引擎和训练引擎底层有着不同的优化方法，代码走不同的分支，两者都可以进行预测。在入门教程的训练/评估/预测流程中介绍的预测流程，即tools/infer.py是使用训练引擎分支的预测流程。保存的inference_model，可以通过`fluid.io.load_inference_model`接口，走训练引擎分支预测。本文档也同时介绍通过预测引擎的Python API进行预测，一般而言这种方式的速度优于前者。


这篇教程介绍的Python API预测示例，除了可视化部分依赖PaddleDetection外，预处理、模型结构、执行流程均不依赖PaddleDetection。


## 使用方式

```bash
export CUDA_VISIBLE_DEVICES=0
python tools/cpp_infer.py --model_path=inference_model/faster_rcnn_r50_1x/ --config_path=tools/cpp_demo.yml --infer_img=demo/000000570688.jpg --visualize
```


主要参数说明：

-  model_path:  inference_model保存路径
-  config_path: 参数配置、数据预处理配置文件，注意不是训练时的配置文件
-  infer_img:   待预测图片
-  visualize:   是否保存可视化结果，默认保存路径为```output/```


更多参数可在```tools/cpp_demo.yml```中查看，主要参数：


- use_python_inference:

  若为true，使用`fluid.io.load_inference_model`接口，走训练引擎分支预测。

- mode:

  支持fluid、trt_fp32、trt_fp16、trt_int8，当use_python_inference为false时起作用。fluid是通过预测引擎分支预测，trt_fp32、trt_fp16、trt_int8是通过预测引擎分支预测，后端基于TensorRT的FP32、FP16精度。

- min_subgraph_size:

  当设置mode采用TensorRT时，注意设置此参数。设置与模型arch相关，对部分arch需要调大该参数，一般设置为40适用于所有模型。适当的调小`min_subgraph_size`会对预测有加速效果，例如YOLO中该参数可设置为3。

- Preprocess:

  数据预处理配置，一般来说顺序为Resize -> Normalize -> Permute，对于FPN模型还需配置PadStride。不同模型的数据预处理参考训练配置中的`TestReader`部分。


**注意**

1. 基于TensorRT预测，数据预处理Resize设置的shape必须保持与模型导出时shape大小一致。

2. 预处理中`PadStride`为输入图片右下角填充0，默认设置stride为0，即不对输入图片做padding操作。模型中包含FPN结构时，stride应设置为32。模型为RetinaNet系列模型时，stride应设置为128.

3. PaddlePaddle默认的GPU安装包(<=1.7)，是不支持基于TensorRT进行预测，如果想基于TensorRT加速预测，需要自行编译，详细可参考[预测库编译教程](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_usage/deploy/inference/paddle_tensorrt_infer.html)。
