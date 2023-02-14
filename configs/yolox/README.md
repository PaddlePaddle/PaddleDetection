# YOLOX (YOLOX: Exceeding YOLO Series in 2021)

## 内容
- [模型库](#模型库)
- [使用说明](#使用说明)
- [速度测试](#速度测试)
- [引用](#引用)


## 模型库
### YOLOX on COCO

| 网络网络        | 输入尺寸   | 图片数/GPU | 学习率策略 | 模型推理耗时(ms) | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 | Params(M) | FLOPs(G) |    下载链接       | 配置文件 |
| :------------- | :------- | :-------: | :------: | :------------: | :---------------------: | :----------------: |:---------: | :------: |:---------------: |:-----: |
| YOLOX-nano     |  416     |    8      |   300e    |     2.3    |  26.1  |  42.0 |  0.91  |  1.08 | [下载链接](https://paddledet.bj.bcebos.com/models/yolox_nano_300e_coco.pdparams) | [配置文件](./yolox_nano_300e_coco.yml) |
| YOLOX-tiny     |  416     |    8      |   300e    |     2.8    |  32.9  |  50.4 |  5.06  |  6.45 | [下载链接](https://paddledet.bj.bcebos.com/models/yolox_tiny_300e_coco.pdparams) | [配置文件](./yolox_tiny_300e_coco.yml) |
| YOLOX-s        |  640     |    8      |   300e    |     3.0    |  40.4  |  59.6 |  9.0  |  26.8 | [下载链接](https://paddledet.bj.bcebos.com/models/yolox_s_300e_coco.pdparams) | [配置文件](./yolox_s_300e_coco.yml) |
| YOLOX-m        |  640     |    8      |   300e    |     5.8    |  46.9  |  65.7 |  25.3  |  73.8 | [下载链接](https://paddledet.bj.bcebos.com/models/yolox_m_300e_coco.pdparams) | [配置文件](./yolox_m_300e_coco.yml) |
| YOLOX-l        |  640     |    8      |   300e    |     9.3    |  50.1  |  68.8 |  54.2  |  155.6 | [下载链接](https://paddledet.bj.bcebos.com/models/yolox_l_300e_coco.pdparams) | [配置文件](./yolox_l_300e_coco.yml) |
| YOLOX-x        |  640     |    8      |   300e    |     16.6   |  **51.8**  |  **70.6** |  99.1  |  281.9 | [下载链接](https://paddledet.bj.bcebos.com/models/yolox_x_300e_coco.pdparams) | [配置文件](./yolox_x_300e_coco.yml) |


| 网络网络        | 输入尺寸   | 图片数/GPU | 学习率策略 | 模型推理耗时(ms) | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 | Params(M) | FLOPs(G) |    下载链接       | 配置文件 |
| :------------- | :------- | :-------: | :------: | :------------: | :---------------------: | :----------------: |:---------: | :------: |:---------------: |:-----: |
| YOLOX-cdn-tiny    |  416     |    8      |   300e    |     1.9    |  32.4  |  50.2 |  5.03 |  6.33  | [下载链接](https://paddledet.bj.bcebos.com/models/yolox_cdn_tiny_300e_coco.pdparams) | [配置文件](./yolox_cdn_tiny_300e_coco.yml) |
| YOLOX-crn-s     |  640     |    8      |   300e    |     3.0    |  40.4  |  59.6 |  7.7  |  24.69 | [下载链接](https://paddledet.bj.bcebos.com/models/yolox_crn_s_300e_coco.pdparams) | [配置文件](./yolox_crn_s_300e_coco.yml) |
| YOLOX-ConvNeXt-s|  640     |    8      |   36e     |     -      |  **44.6**  |  **65.3** |  36.2 |  27.52 | [下载链接](https://paddledet.bj.bcebos.com/models/yolox_convnext_s_36e_coco.pdparams) | [配置文件](../convnext/yolox_convnext_s_36e_coco.yml) |


**注意:**
  - YOLOX模型训练使用COCO train2017作为训练集，YOLOX-cdn表示使用与YOLOv5 releases v6.0之后版本相同的主干网络，YOLOX-crn表示使用与PPYOLOE相同的主干网络CSPResNet，YOLOX-ConvNeXt表示使用ConvNeXt作为主干网络；
  - YOLOX模型训练过程中默认使用8 GPUs进行混合精度训练，默认每卡batch_size为8，默认lr为0.01为8卡总batch_size=64的设置，如果**GPU卡数**或者每卡**batch size**发生了改变，你需要按照公式 **lr<sub>new</sub> = lr<sub>default</sub> * (batch_size<sub>new</sub> * GPU_number<sub>new</sub>) / (batch_size<sub>default</sub> * GPU_number<sub>default</sub>)** 调整学习率；
  - 为保持高mAP的同时提高推理速度，可以将[yolox_cspdarknet.yml](_base_/yolox_cspdarknet.yml)中的`nms_top_k`修改为`1000`，将`keep_top_k`修改为`100`，将`score_threshold`修改为`0.01`，mAP会下降约0.1~0.2%；
  - 为快速的demo演示效果，可以将[yolox_cspdarknet.yml](_base_/yolox_cspdarknet.yml)中的`score_threshold`修改为`0.25`，将`nms_threshold`修改为`0.45`，但mAP会下降较多；
  - YOLOX模型推理速度测试采用单卡V100，batch size=1进行测试，使用**CUDA 10.2**, **CUDNN 7.6.5**，TensorRT推理速度测试使用**TensorRT 6.0.1.8**。
  - 参考[速度测试](#速度测试)以复现YOLOX推理速度测试结果，速度为**tensorRT-FP16**测速后的最快速度，**不包含数据预处理和模型输出后处理(NMS)**的耗时。
  - 如果你设置了`--run_benchmark=True`, 你首先需要安装以下依赖`pip install pynvml psutil GPUtil`。

## 使用教程

### 1.训练
执行以下指令使用混合精度训练YOLOX
```bash
python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/yolox/yolox_s_300e_coco.yml --amp --eval
```
**注意:**
- `--amp`表示开启混合精度训练以避免显存溢出，`--eval`表示边训边验证。

### 2.评估
执行以下命令在单个GPU上评估COCO val2017数据集
```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/yolox/yolox_s_300e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/yolox_s_300e_coco.pdparams
```

### 3.推理
使用以下命令在单张GPU上预测图片，使用`--infer_img`推理单张图片以及使用`--infer_dir`推理文件中的所有图片。
```bash
# 推理单张图片
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/yolox/yolox_s_300e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/yolox_s_300e_coco.pdparams --infer_img=demo/000000014439_640x640.jpg

# 推理文件中的所有图片
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/yolox/yolox_s_300e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/yolox_s_300e_coco.pdparams --infer_dir=demo
```

### 4.导出模型
YOLOX在GPU上推理部署或benchmark测速等需要通过`tools/export_model.py`导出模型。

当你**使用Paddle Inference但不使用TensorRT**时，运行以下的命令导出模型

```bash
python tools/export_model.py -c configs/yolox/yolox_s_300e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/yolox_s_300e_coco.pdparams
```

当你**使用Paddle Inference且使用TensorRT**时，需要指定`-o trt=True`来导出模型。

```bash
python tools/export_model.py -c configs/yolox/yolox_s_300e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/yolox_s_300e_coco.pdparams trt=True
```

如果你想将YOLOX模型导出为**ONNX格式**，参考
[PaddleDetection模型导出为ONNX格式教程](../../deploy/EXPORT_ONNX_MODEL.md)，运行以下命令：

```bash

# 导出推理模型
python tools/export_model.py -c configs/yolox/yolox_s_300e_coco.yml --output_dir=output_inference -o weights=https://paddledet.bj.bcebos.com/models/yolox_s_300e_coco.pdparams

# 安装paddle2onnx
pip install paddle2onnx

# 转换成onnx格式
paddle2onnx --model_dir output_inference/yolox_s_300e_coco --model_filename model.pdmodel --params_filename model.pdiparams --opset_version 11 --save_file yolox_s_300e_coco.onnx
```

**注意：** ONNX模型目前只支持batch_size=1


### 5.推理部署
YOLOX可以使用以下方式进行部署：
  - Paddle Inference [Python](../../deploy/python) & [C++](../../deploy/cpp)
  - [Paddle-TensorRT](../../deploy/TENSOR_RT.md)
  - [PaddleServing](https://github.com/PaddlePaddle/Serving)
  - [PaddleSlim模型量化](../slim)

运行以下命令导出模型

```bash
python tools/export_model.py -c configs/yolox/yolox_s_300e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/yolox_s_300e_coco.pdparams trt=True
```

**注意：**
- trt=True表示**使用Paddle Inference且使用TensorRT**进行测速，速度会更快，默认不加即为False，表示**使用Paddle Inference但不使用TensorRT**进行测速。
- 如果是使用Paddle Inference在TensorRT FP16模式下部署，需要参考[Paddle Inference文档](https://www.paddlepaddle.org.cn/inference/master/user_guides/download_lib.html#python)，下载并安装与你的CUDA, CUDNN和TensorRT相应的wheel包。

#### 5.1.Python部署
`deploy/python/infer.py`使用上述导出后的Paddle Inference模型用于推理和benchnark测速，如果设置了`--run_benchmark=True`, 首先需要安装以下依赖`pip install pynvml psutil GPUtil`。

```bash
# Python部署推理单张图片
python deploy/python/infer.py --model_dir=output_inference/yolox_s_300e_coco --image_file=demo/000000014439_640x640.jpg --device=gpu

# 推理文件夹下的所有图片
python deploy/python/infer.py --model_dir=output_inference/yolox_s_300e_coco --image_dir=demo/ --device=gpu
```

#### 5.2. C++部署
`deploy/cpp/build/main`使用上述导出后的Paddle Inference模型用于C++推理部署, 首先按照[docs](../../deploy/cpp/docs)编译安装环境。
```bash
# C++部署推理单张图片
./deploy/cpp/build/main --model_dir=output_inference/yolox_s_300e_coco/ --image_file=demo/000000014439_640x640.jpg --run_mode=paddle --device=GPU --threshold=0.5 --output_dir=cpp_infer_output/yolox_s_300e_coco
```


## 速度测试

为了公平起见，在[模型库](#模型库)中的速度测试结果均为不包含数据预处理和模型输出后处理(NMS)的数据(与[YOLOv4(AlexyAB)](https://github.com/AlexeyAB/darknet)测试方法一致)，需要在导出模型时指定`-o exclude_nms=True`。测速需设置`--run_benchmark=True`, 首先需要安装以下依赖`pip install pynvml psutil GPUtil`。

**使用Paddle Inference但不使用TensorRT**进行测速，执行以下命令：

```bash
# 导出模型
python tools/export_model.py -c configs/yolox/yolox_s_300e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/yolox_s_300e_coco.pdparams exclude_nms=True

# 速度测试，使用run_benchmark=True
python deploy/python/infer.py --model_dir=output_inference/yolox_s_300e_coco --image_file=demo/000000014439_640x640.jpg --run_mode=paddle --device=gpu --run_benchmark=True
```

**使用Paddle Inference且使用TensorRT**进行测速，执行以下命令：

```bash
# 导出模型，使用trt=True
python tools/export_model.py -c configs/yolox/yolox_s_300e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/yolox_s_300e_coco.pdparams exclude_nms=True trt=True

# 速度测试，使用run_benchmark=True
python deploy/python/infer.py --model_dir=output_inference/yolox_s_300e_coco --image_file=demo/000000014439_640x640.jpg --device=gpu --run_benchmark=True

# tensorRT-FP32测速
python deploy/python/infer.py --model_dir=output_inference/yolox_s_300e_coco --image_file=demo/000000014439_640x640.jpg --device=gpu --run_benchmark=True --run_mode=trt_fp32

# tensorRT-FP16测速
python deploy/python/infer.py --model_dir=output_inference/yolox_s_300e_coco --image_file=demo/000000014439_640x640.jpg --device=gpu --run_benchmark=True --run_mode=trt_fp16
```
**注意:**
- 导出模型时指定`-o exclude_nms=True`仅作为测速时用，这样导出的模型其推理部署预测的结果不是最终检出框的结果。
- [模型库](#模型库)中的速度测试结果为**tensorRT-FP16**测速后的最快速度，为**不包含数据预处理和模型输出后处理(NMS)**的耗时。

## FAQ

<details>
<summary>如何计算模型参数量</summary>
可以将以下代码插入：[trainer.py](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.6/ppdet/engine/trainer.py#L154) 来计算参数量。
```python
params = sum([
    p.numel() for n, p in self.model.named_parameters()
    if all([x not in n for x in ['_mean', '_variance']])
]) # exclude BatchNorm running status
print('Params: ', params)
```
</details>


## Citations
```
 @article{yolox2021,
  title={YOLOX: Exceeding YOLO Series in 2021},
  author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2107.08430},
  year={2021}
}
```
