# YOLOX (YOLOX: Exceeding YOLO Series in 2021)

## Model Zoo
### YOLOX on COCO

| 网络网络        | 输入尺寸   | 图片数/GPU | 学习率策略 |推理时间(fps) | Box AP |    下载链接       | 配置文件 |
| :------------- | :------- | :-------: | :------: | :---------: | :-----: | :-------------: | :-----: |
| YOLOX-tiny     |  416     |    8      |   300e    |     ----    |  32.9  | [下载链接](https://paddledet.bj.bcebos.com/models/yolox_tiny_300e_coco.pdparams) | [配置文件](./yolox_tiny_300e_coco.yml) |

## 使用教程

### 1. 训练
执行以下指令使用混合精度训练YOLOX
```bash
python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/yolox/yolox_s_300e_coco.yml --fleet --amp --eval
```
** 注意: ** 使用默认配置训练需要设置`--fleet`, `--amp`最好也设置以避免显存溢出.

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
YOLOX在GPU上部署或者推理benchmark需要通过`tools/export_model.py`导出模型。
当你使用Paddle Inference但不使用TensorRT时，运行以下的命令进行导出：
```bash
python tools/export_model.py -c configs/yolox/yolox_s_300e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/yolox_s_300e_coco.pdparams
```
`deploy/python/infer.py`使用上述导出后的Paddle Inference模型用于推理和benchnark.

```bash
# 推理单张图片
CUDA_VISIBLE_DEVICES=0 python deploy/python/infer.py --model_dir=output_inference/yolox_s_300e_coco --image_file=demo/000000014439_640x640.jpg --device=gpu

# 推理文件夹下的所有图片
CUDA_VISIBLE_DEVICES=0 python deploy/python/infer.py --model_dir=output_inference/yolox_s_300e_coco --image_dir=demo/ --device=gpu

# benchmark
CUDA_VISIBLE_DEVICES=0 python deploy/python/infer.py --model_dir=output_inference/yolox_s_300e_coco --image_file=demo/000000014439_640x640.jpg --device=gpu --run_benchmark=True
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
