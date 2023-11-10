# PPYOLOE-Inst


## 简介
PPYOLOE-Inst是一个实例分割模型。

## 模型库
|     Model      | Epoch | Input shape | Box AP | Mask AP | Params(M) | FLOPs(G) | T4 TensorRT FP16(FPS) | Pretrained Model |                   config                   |
|:--------------:|:-----:|:-----------:|:------:|:-------:|:---------:|:--------:|:---------------------:|:----------------:|:------------------------------------------:|
| PPYOLOE-Inst-L |  300  |     640     |        |         |           |          |                       |                  | [config](ppyoloe_inst_crn_l_300e_coco.yml) |


## 快速开始

<details open>
<summary>依赖包:</summary>

- PaddlePaddle >= 2.4.1

</details>

<details>
<summary>安装</summary>

- [安装指导文档](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/docs/tutorials/INSTALL.md)

</details>

<details>
<summary>训练&评估</summary>

- 单卡GPU上训练:

```shell
# training on single-GPU
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/ppyoloe_inst/ppyoloe_inst_crn_l_300e_coco.yml --eval
```

- 多卡GPU上训练:

```shell
# training on multi-GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/ppyoloe_inst/ppyoloe_inst_crn_l_300e_coco.yml --eval --amp
```

- 评估:

```shell
python tools/eval.py -c configs/ppyoloe_inst/ppyoloe_inst_crn_l_300e_coco.yml \
              -o weights=${model_params_path}
```
