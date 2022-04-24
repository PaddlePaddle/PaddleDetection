# YOLOX (YOLOX: Exceeding YOLO Series in 2021)

## Model Zoo
### YOLOX on COCO

| 网络网络        | 输入尺寸   | 图片数/GPU | 学习率策略 |推理时间(fps) | Box AP |    下载链接       | 配置文件 |
| :------------- | :------- | :-------: | :------: | :---------: | :-----: | :-------------: | :-----: |
| YOLOX-nano     |  416     |    8      |   300e    |     ----    |  26.1  | [下载链接](https://paddledet.bj.bcebos.com/models/yolox_nano_300e_coco.pdparams) | [配置文件](./yolox_nano_300e_coco.yml) |
| YOLOX-tiny     |  416     |    8      |   300e    |     ----    |  32.9  | [下载链接](https://paddledet.bj.bcebos.com/models/yolox_tiny_300e_coco.pdparams) | [配置文件](./yolox_tiny_300e_coco.yml) |
| YOLOX-s        |  640     |    8      |   300e    |     ----    |  40.4  | [下载链接](https://paddledet.bj.bcebos.com/models/yolox_s_300e_coco.pdparams) | [配置文件](./yolox_s_300e_coco.yml) |
| YOLOX-m        |  640     |    8      |   300e    |     ----    |  46.9  | [下载链接](https://paddledet.bj.bcebos.com/models/yolox_m_300e_coco.pdparams) | [配置文件](./yolox_m_300e_coco.yml) |
| YOLOX-l        |  640     |    8      |   300e    |     ----    |  50.1  | [下载链接](https://paddledet.bj.bcebos.com/models/yolox_l_300e_coco.pdparams) | [配置文件](./yolox_l_300e_coco.yml) |
| YOLOX-x        |  640     |    8      |   300e    |     ----    |  51.8  | [下载链接](https://paddledet.bj.bcebos.com/models/yolox_x_300e_coco.pdparams) | [配置文件](./yolox_x_300e_coco.yml) |

**注意:**
  - YOLOX模型训练使用COCO train2017作为训练集，Box AP为在COCO val2017上的`mAP(IoU=0.5:0.95)`结果；
  - YOLOX模型训练过程中默认使用8 GPUs进行混合精度训练，总batch_size为64，如果**GPU卡数**或者**batch size**发生了改变，你需要按照公式 **lr<sub>new</sub> = lr<sub>default</sub> * (batch_size<sub>new</sub> * GPU_number<sub>new</sub>) / (batch_size<sub>default</sub> * GPU_number<sub>default</sub>)** 调整学习率；
  - 为保持高mAP的同时提高推理速度，可以将[yolox_cspdarknet.yml](_base_/yolox_cspdarknet.yml)中的`nms_top_k`修改为`1000`，将`keep_top_k`修改为`100`，mAP会下降约0.1~0.2%；
  - 为快速的demo演示效果，可以将[yolox_cspdarknet.yml](_base_/yolox_cspdarknet.yml)中的`score_threshold`修改为`0.25`，将`nms_threshold`修改为`0.45`，但mAP会下降较多；


## 使用教程

### 1. 训练
执行以下指令使用混合精度训练YOLOX
```bash
python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/yolox/yolox_s_300e_coco.yml --fleet --amp --eval
```
**注意:**
使用默认配置训练需要设置`--fleet`，`--amp`最好也设置以避免显存溢出，`--eval`表示边训边验证。

### 2. 评估
执行以下命令在单个GPU上评估COCO val2017数据集
```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/yolox/yolox_s_300e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/yolox_s_300e_coco.pdparams
```

### 3. 推理
使用以下命令在单张GPU上预测图片，使用`--infer_img`推理单张图片以及使用`--infer_dir`推理文件中的所有图片。
```bash
# 推理单张图片
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/yolox/yolox_s_300e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/yolox_s_300e_coco.pdparams --infer_img=demo/000000014439_640x640.jpg

# 推理文件中的所有图片
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/yolox/yolox_s_300e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/yolox_s_300e_coco.pdparams --infer_dir=demo
```

### 4. 部署

#### 4.1. 导出模型
YOLOX在GPU上推理部署或benchmark测速等需要通过`tools/export_model.py`导出模型。
运行以下的命令进行导出：
```bash
python tools/export_model.py -c configs/yolox/yolox_s_300e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/yolox_s_300e_coco.pdparams
```

#### 4.2. Python部署
`deploy/python/infer.py`使用上述导出后的Paddle Inference模型用于推理和benchnark测速，如果设置了`--run_benchmark=True`, 首先需要安装以下依赖`pip install pynvml psutil GPUtil`。

```bash
# Python部署推理单张图片
python deploy/python/infer.py --model_dir=output_inference/yolox_s_300e_coco --image_file=demo/000000014439_640x640.jpg --device=gpu

# 推理文件夹下的所有图片
python deploy/python/infer.py --model_dir=output_inference/yolox_s_300e_coco --image_dir=demo/ --device=gpu

# benchmark测速
python deploy/python/infer.py --model_dir=output_inference/yolox_s_300e_coco --image_file=demo/000000014439_640x640.jpg --device=gpu --run_benchmark=True

# tensorRT-FP32测速
python deploy/python/infer.py --model_dir=output_inference/yolox_s_300e_coco --image_file=demo/000000014439_640x640.jpg --device=gpu --run_benchmark=True --trt_max_shape=640 --trt_min_shape=640 --trt_opt_shape=640 --run_mode=trt_fp32

# tensorRT-FP16测速
python deploy/python/infer.py --model_dir=output_inference/yolox_s_300e_coco --image_file=demo/000000014439_640x640.jpg --device=gpu --run_benchmark=True --trt_max_shape=640 --trt_min_shape=640 --trt_opt_shape=640 --run_mode=trt_fp16
```

#### 4.2. C++部署
`deploy/cpp/build/main`使用上述导出后的Paddle Inference模型用于C++推理部署, 首先按照[docs](../../deploy/cpp/docs)编译安装环境。
```bash
# C++部署推理单张图片
./deploy/cpp/build/main --model_dir=output_inference/yolox_s_300e_coco/ --image_file=demo/000000014439_640x640.jpg --run_mode=paddle --device=GPU --threshold=0.5 --output_dir=cpp_infer_output/yolox_s_300e_coco
```

## Citations
```
 @article{yolox2021,
  title={YOLOX: Exceeding YOLO Series in 2021},
  author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2107.08430},
  year={2021}
}
```
