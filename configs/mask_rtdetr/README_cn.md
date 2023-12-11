简体中文 | [English](README.md)

# Mask RT-DETR

## 内容
- [简介](#简介)
- [模型库](#模型库)
- [使用说明](#使用说明)
- [更多用法](#更多用法)

## 简介
Mask-RT-DETR是[RT-DETR](../rtdetr/README.md)的实例分割版本。

## 模型库
|     Model      | Epoch | Backbone | Input shape | Box AP | Mask AP | Params(M) | FLOPs(G) | T4 TensorRT FP16(FPS) |                                    Pretrained Model                                    |                   config                    |
|:--------------:|:-----:|:--------:|:-----------:|:------:|:-------:|:---------:|:--------:|:---------------------:|:--------------------------------------------------------------------------------------:|:-------------------------------------------:|
| Mask-RT-DETR-L |  6x   | HGNetv2  |     640     |  51.3  |  45.7   |    32     |   120    |          90           | [model](https://paddledet.bj.bcebos.com/models/mask_rtdetr_hgnetv2_l_6x_coco.pdparams) | [config](mask_rtdetr_hgnetv2_l_6x_coco.yml) |


## 使用说明

### 数据集和评价指标

下载PaddleDetection团队提供的**COCO数据**，并解压放置于`PaddleDetection/dataset/`下：

```
wget https://bj.bcebos.com/v1/paddledet/data/coco.tar
# tar -xvf coco.tar
```

**注意:**
 - COCO风格格式，请参考 [format-data](https://cocodataset.org/#format-data) 和 [format-results](https://cocodataset.org/#format-results)。
 - COCO风格评测指标，请参考 [detection-eval](https://cocodataset.org/#detection-eval) ，并首先安装 [cocoapi](https://github.com/cocodataset/cocoapi)。

### 自定义数据集

1.自定义数据集的标注制作，请参考 [DetAnnoTools](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/docs/tutorials/data/DetAnnoTools.md);
2.自定义数据集的训练准备，请参考 [PrepareDataSet](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/docs/tutorials/data/PrepareDetDataSet.md).


### 训练

请执行以下指令训练Mask RT-DETR

```bash
# 单卡GPU上训练
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/mask_rtdetr/mask_rtdetr_hgnetv2_l_6x_coco.yml --eval

# 多卡GPU上训练
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/mask_rtdetr/mask_rtdetr_hgnetv2_l_6x_coco.yml --amp --eval
```
**注意:**
- 如果需要边训练边评估，请添加`--eval`.
- Mask RT-DETR支持混合精度训练，请添加`--amp`.
- PaddleDetection支持多机训练，可以参考[多机训练教程](../../docs/tutorials/DistributedTraining_cn.md).

### 评估

执行以下命令在单个GPU上评估COCO val2017数据集

```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/mask_rtdetr/mask_rtdetr_hgnetv2_l_6x_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/mask_rtdetr_hgnetv2_l_6x_coco.pdparams
```

在coco test-dev2017上评估，请先从[COCO数据集下载](https://cocodataset.org/#download)下载COCO test-dev2017数据集，然后解压到COCO数据集文件夹并像`configs/ppyolo/ppyolo_test.yml`一样配置`EvalDataset`。

### 推理

使用以下命令在单张GPU上预测图片，使用`--infer_img`推理单张图片以及使用`--infer_dir`推理文件中的所有图片。


```bash
# 推理单张图片
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/mask_rtdetr/mask_rtdetr_hgnetv2_l_6x_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/mask_rtdetr_hgnetv2_l_6x_coco.pdparams --infer_img=demo/000000014439_640x640.jpg

# 推理文件中的所有图片
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/mask_rtdetr/mask_rtdetr_hgnetv2_l_6x_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/mask_rtdetr_hgnetv2_l_6x_coco.pdparams --infer_dir=demo
```

### 模型导出

Mask RT-DETR在GPU上部署或者速度测试需要通过`tools/export_model.py`导出模型。

当你**使用Paddle Inference但不使用TensorRT**时，运行以下的命令导出模型

```bash
python tools/export_model.py -c configs/mask_rtdetr/mask_rtdetr_hgnetv2_l_6x_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/mask_rtdetr_hgnetv2_l_6x_coco.pdparams
```

当你**使用Paddle Inference且使用TensorRT**时，需要指定`-o trt=True`来导出模型。

```bash
python tools/export_model.py -c configs/mask_rtdetr/mask_rtdetr_hgnetv2_l_6x_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/mask_rtdetr_hgnetv2_l_6x_coco.pdparams trt=True
```

如果你想将PP-YOLOE模型导出为**ONNX格式**，参考
[PaddleDetection模型导出为ONNX格式教程](../../deploy/EXPORT_ONNX_MODEL.md)，运行以下命令：

```bash

# 导出推理模型
python tools/export_model.py -c configs/mask_rtdetr/mask_rtdetr_hgnetv2_l_6x_coco.yml --output_dir=output_inference -o weights=https://paddledet.bj.bcebos.com/models/mask_rtdetr_hgnetv2_l_6x_coco.pdparams trt=True

# 安装paddle2onnx
pip install paddle2onnx

# 转换成onnx格式
paddle2onnx --model_dir output_inference/mask_rtdetr_hgnetv2_l_6x_coco --model_filename model.pdmodel --params_filename model.pdiparams --opset_version 16 --save_file mask_rtdetr_hgnetv2_l_6x_coco.onnx
```

**注意：** ONNX模型目前只支持batch_size=1

### 速度测试

**使用 ONNX 和 TensorRT** 进行测速，执行以下命令：

```bash
# 导出模型
python tools/export_model.py -c configs/mask_rtdetr/mask_rtdetr_hgnetv2_l_6x_coco.yml --output_dir=output_inference -o weights=https://paddledet.bj.bcebos.com/models/mask_rtdetr_hgnetv2_l_6x_coco.pdparams trt=True

# 转化成ONNX格式
paddle2onnx --model_dir output_inference/mask_rtdetr_hgnetv2_l_6x_coco --model_filename model.pdmodel --params_filename model.pdiparams --opset_version 16 --save_file mask_rtdetr_hgnetv2_l_6x_coco.onnx

```

固定先前导出的ONNX模型的`im_shape`和`scale_factor`两个输入数据，代码如下：
```python
# onnx_edit.py

import copy

import onnx

if __name__ == '__main__':
    model_path = './mask_rtdetr_hgnetv2_l_6x_coco.onnx'
    model = onnx.load_model(model_path)

    im_shape = onnx.helper.make_tensor(
        name='im_shape',
        data_type=onnx.helper.TensorProto.FLOAT,
        dims=[1, 2],
        vals=[640, 640])
    scale_factor = onnx.helper.make_tensor(
        name='scale_factor',
        data_type=onnx.helper.TensorProto.FLOAT,
        dims=[1, 2],
        vals=[1, 1])

    new_model = copy.deepcopy(model)

    for input in model.graph.input:
        if input.name == 'im_shape':
            new_model.graph.input.remove(input)
            new_model.graph.initializer.append(im_shape)

        if input.name == 'scale_factor':
            new_model.graph.input.remove(input)
            new_model.graph.initializer.append(scale_factor)

    onnx.checker.check_model(new_model, full_check=True)
    onnx.save_model(new_model, model_path)
```

使用onnxsim简化onnx模型：
```shell
pip install onnxsim
onnxsim mask_rtdetr_hgnetv2_l_6x_coco.onnx mask_rtdetr_hgnetv2_l_6x_coco.onnx --overwrite-input-shape "image:1,3,640,640"
```

```shell
# 测试速度，半精度，batch_size=1
trtexec --onnx=./mask_rtdetr_hgnetv2_l_6x_coco.onnx --saveEngine=./mask_rtdetr_hgnetv2_l_6x_coco.engine --workspace=4096 --avgRuns=1000 --fp16
```


### 部署

Mask RT-DETR可以使用以下方式进行部署：
  - Paddle Inference [Python](../../deploy/python) & [C++](../../deploy/cpp)

接下来，我们将介绍Mask RT-DETR如何使用Paddle Inference在TensorRT FP16模式下部署

首先，参考[Paddle Inference文档](https://www.paddlepaddle.org.cn/inference/master/user_guides/download_lib.html#python)，下载并安装与你的CUDA, CUDNN和TensorRT相应的wheel包。

然后，运行以下命令导出模型

```bash
python tools/export_model.py -c configs/mask_rtdetr/mask_rtdetr_hgnetv2_l_6x_coco.yml --output_dir=output_inference -o weights=https://paddledet.bj.bcebos.com/models/mask_rtdetr_hgnetv2_l_6x_coco.pdparams trt=True
```

最后，使用TensorRT FP16进行推理

```bash
# 推理单张图片
CUDA_VISIBLE_DEVICES=0 python deploy/python/infer.py --model_dir=output_inference/mask_rtdetr_hgnetv2_l_6x_coco --image_file=demo/000000014439_640x640.jpg --device=gpu --run_mode=trt_fp16

# 推理文件夹下的所有图片
CUDA_VISIBLE_DEVICES=0 python deploy/python/infer.py --model_dir=output_inference/mask_rtdetr_hgnetv2_l_6x_coco --image_dir=demo/ --device=gpu  --run_mode=trt_fp16

```

**注意：**
- TensorRT会根据网络的定义，执行针对当前硬件平台的优化，生成推理引擎并序列化为文件。该推理引擎只适用于当前软硬件平台。如果你的软硬件平台没有发生变化，你可以设置[enable_tensorrt_engine](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/deploy/python/infer.py#L660)的参数`use_static=True`，这样生成的序列化文件将会保存在`output_inference`文件夹下，下次执行TensorRT时将加载保存的序列化文件。


## 更多用法

### 模型权重保存指标的设定
在实例分割任务中，如果使用COCO风格的评测指标，且推理结果中同时包含`box`和`mask`, 那么默认的模型权重保存指标是`box ap`。如果想要将保存指标改为`mask ap`，可以在配置文件中添加`target_metrics`参数，并将其赋值为`mask`。具体的配置文件示例可以参考[mask_rtdetr_r50vd.yml](./_base_/mask_rtdetr_r50vd.yml)。

### Query数量的设定

- 后处理中Query数量的设定可以通过修改`DETRPostProcess`的`num_top_queries`参数来调整。Mask RT-DETR后处理的`num_top_queries`默认值为100。
- Query选择部分Query数量的设定可以通过修改`MaskRTDETR`的`num_queries`参数来调整。Mask RT-DETR的`num_queries`默认值为300。由于Mask RT-DETR的分类损失考虑了mask IoU和置信度的一致性，我们可以将Mask RT-DETR的`num_queries`设置为100，并直接加载在`num_queries`为300时训练的权重。这样可以在精度损失不大的情况下，提高推理速度。
