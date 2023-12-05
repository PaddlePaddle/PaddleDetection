English | [简体中文](README_cn.md)

# Mask RT-DETR

## Table of Contents
- [Introduction](#Introduction)
- [Model Zoo](#Model Zoo)
- [Getting Start](#Getting Start)

## Introduction
Mask RT-DETR is an instance segmentation version of [RT DETR](../rtdetr/README.md).

## Model Zoo
|        Model        | Epoch | Backbone | Input shape | Box AP | Mask AP | Params(M) | FLOPs(G) | T4 TensorRT FP16(FPS) | Pretrained Model |                      config                      |
|:-------------------:|:-----:|:--------:|:-----------:|:------:|:-------:|:---------:|:--------:|:---------------------:|:----------------:|:------------------------------------------------:|
|   Mask-RT-DETR-L    |  6x   | HGNetv2  |     640     |        |  45.7   |    32     |   120    |          90           |                  |   [config](mask_rtdetr_hgnetv2_l_6x_coco.yml)    |


## Getting Start

### Datasets and Metrics

PaddleDetection team provides **COCO dataset** , decompress and place it under `PaddleDetection/dataset/`:

```
wget https://bj.bcebos.com/v1/paddledet/data/coco.tar
# tar -xvf coco.tar
```

**Note:**
  - For the format of COCO style dataset, please refer to [format-data](https://cocodataset.org/#format-data) and [format-results](https://cocodataset.org/#format-results).
  - For the evaluation metric of COCO, please refer to [detection-eval](https://cocodataset.org/#detection-eval), and install  [cocoapi](https://github.com/cocodataset/cocoapi) at first.

### Custom dataset

1.For the annotation of custom dataset, please refer to [DetAnnoTools](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/docs/tutorials/data/DetAnnoTools_en.md);

2.For training preparation of custom dataset，please refer to [PrepareDataSet](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/docs/tutorials/data/PrepareDetDataSet_en.md).


### Training

Training Mask RT-DETR with following command

```bash
# training on a single GPU
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/mask_rtdetr/mask_rtdetr_hgnetv2_l_6x_coco.yml --eval

# training on multi GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/mask_rtdetr/mask_rtdetr_hgnetv2_l_6x_coco.yml --amp --eval
```
**Notes:**
- If you need to evaluate while training, please add `--eval`.
- Mask RT-DETR supports mixed precision training, please add `--amp`.
- PaddleDetection supports multi-machine distributed training, you can refer to [DistributedTraining tutorial](../../docs/tutorials/DistributedTraining_en.md).


### Evaluation

Evaluating Mask RT-DETR on COCO val2017 dataset in single GPU with following commands:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/mask_rtdetr/mask_rtdetr_hgnetv2_l_6x_coco.yml -o weights=${model_params_path}
```

For evaluation on COCO test-dev2017 dataset, please download COCO test-dev2017 dataset from [COCO dataset download](https://cocodataset.org/#download) and decompress to COCO dataset directory and configure `EvalDataset` like `configs/ppyolo/ppyolo_test.yml`.

### Inference

Inference images in single GPU with following commands, use `--infer_img` to inference a single image and `--infer_dir` to inference all images in the directory.


```bash
# inference single image
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/mask_rtdetr/mask_rtdetr_hgnetv2_l_6x_coco.yml -o weights=${model_params_path} --infer_img=demo/000000014439_640x640.jpg

# inference all images in the directory
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/mask_rtdetr/mask_rtdetr_hgnetv2_l_6x_coco.yml -o weights=${model_params_path} --infer_dir=demo
```

### Exporting models

For deployment on GPU or speed testing, model should be first exported to inference model using `tools/export_model.py`.

**Exporting Mask RT-DETR for Paddle Inference without TensorRT**, use following command

```bash
python tools/export_model.py -c configs/mask_rtdetr/mask_rtdetr_hgnetv2_l_6x_coco.yml -o weights=${model_params_path}
```

**Exporting Mask RT-DETR for Paddle Inference with TensorRT** for better performance, use following command with extra `-o trt=True` setting.

```bash
python tools/export_model.py -c configs/mask_rtdetr/mask_rtdetr_hgnetv2_l_6x_coco.yml -o weights=${model_params_path} trt=True
```

If you want to export Mask RT-DETR model to **ONNX format**, use following command refer to [PaddleDetection Model Export as ONNX Format Tutorial](../../deploy/EXPORT_ONNX_MODEL_en.md).

```bash

# export inference model
python tools/export_model.py -c configs/mask_rtdetr/mask_rtdetr_hgnetv2_l_6x_coco.yml --output_dir=output_inference -o weights=${model_params_path} trt=True

# install paddle2onnx
pip install paddle2onnx

# convert to onnx
paddle2onnx --model_dir output_inference/mask_rtdetr_hgnetv2_l_6x_coco --model_filename model.pdmodel --params_filename model.pdiparams --opset_version 16 --save_file mask_rtdetr_hgnetv2_l_6x_coco.onnx
```

**Notes:** ONNX model only supports batch_size=1 now

### Speed testing

**Using Paddle Inference with TensorRT** to test speed, run following command

```bash
# export inference model with trt=True
python tools/export_model.py -c configs/mask_rtdetr/mask_rtdetr_hgnetv2_l_6x_coco.yml --output_dir=output_inference -o weights=${model_params_path} trt=True

# convert to onnx
paddle2onnx --model_dir output_inference/mask_rtdetr_hgnetv2_l_6x_coco --model_filename model.pdmodel --params_filename model.pdiparams --opset_version 16 --save_file mask_rtdetr_hgnetv2_l_6x_coco.onnx

```

Fix the previously exported ONNX model's `im_shape` and `scale_factor` two input data, code as follows：
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

    onnx.checker.check_model(model, full_check=True)
    onnx.save_model(new_model, model_path)
```

Simplify the onnx model using onnxsim：

```shell
pip install onnxsim
onnxsim mask_rtdetr_hgnetv2_l_6x_coco.onnx mask_rtdetr_hgnetv2_l_6x_coco.onnx --overwrite-input-shape "image:1,3,640,640"
```

```shell
# trt inference using fp16 and batch_size=1
trtexec --onnx=./mask_rtdetr_hgnetv2_l_6x_coco.onnx --saveEngine=./mask_rtdetr_hgnetv2_l_6x_coco.engine --workspace=4096 --avgRuns=1000 --fp16
```


### Deployment

Mask RT-DETR can be deployed by following approaches:
  - Paddle Inference [Python](../../deploy/python) & [C++](../../deploy/cpp)

Next, we will introduce how to use Paddle Inference to deploy Mask RT-DETR models in TensorRT FP16 mode.

First, refer to [Paddle Inference Docs](https://www.paddlepaddle.org.cn/inference/master/user_guides/download_lib.html#python), download and install packages corresponding to CUDA, CUDNN and TensorRT version.

Then, Exporting Mask RT-DETR for Paddle Inference **with TensorRT**, use following command.

```bash
python tools/export_model.py -c configs/mask_rtdetr/mask_rtdetr_hgnetv2_l_6x_coco.yml --output_dir=output_inference -o weights=${model_params_path} trt=True
```

Finally, inference in TensorRT FP16 mode.

```bash
# inference single image
CUDA_VISIBLE_DEVICES=0 python deploy/python/infer.py --model_dir=output_inference/mask_rtdetr_hgnetv2_l_6x_coco --image_file=demo/000000014439_640x640.jpg --device=gpu --run_mode=trt_fp16

# inference all images in the directory
CUDA_VISIBLE_DEVICES=0 python deploy/python/infer.py --model_dir=output_inference/mask_rtdetr_hgnetv2_l_6x_coco --image_dir=demo/ --device=gpu  --run_mode=trt_fp16

```

**Notes:**
- TensorRT will perform optimization for the current hardware platform according to the definition of the network, generate an inference engine and serialize it into a file. This inference engine is only applicable to the current hardware hardware platform. If your hardware and software platform has not changed, you can set `use_static=True` in [enable_tensorrt_engine](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/deploy/python/infer.py#L660). In this way, the serialized file generated will be saved in the `output_inference` folder, and the saved serialized file will be loaded the next time when TensorRT is executed.
