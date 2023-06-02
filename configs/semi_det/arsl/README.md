简体中文 | [English](README_en.md)

# Ambiguity-Resistant Semi-Supervised Learning for Dense Object Detection (ARSL)

## ARSL-FCOS 模型库

|      模型      |  COCO监督数据比例 | Semi mAP<sup>val<br>0.5:0.95 |  Semi Epochs (Iters)  |  模型下载  |   配置文件   |
| :------------: | :---------:|:----------------------------: | :------------------: |:--------: |:----------: |
| ARSL-FCOS     |    1% |  **22.8**  | 240 (87120)   | [download](https://paddledet.bj.bcebos.com/models/arsl_fcos_r50_fpn_coco_semi001.pdparams) | [config](./arsl_fcos_r50_fpn_coco_semi001.yml) |
| ARSL-FCOS     |    5% |  **33.1**  | 240 (174240)  | [download](https://paddledet.bj.bcebos.com/models/arsl_fcos_r50_fpn_coco_semi005.pdparams) | [config](./arsl_fcos_r50_fpn_coco_semi005.yml ) |
| ARSL-FCOS     |   10% |  **36.9**  | 240 (174240)  | [download](https://paddledet.bj.bcebos.com/models/arsl_fcos_r50_fpn_coco_semi010.pdparams) | [config](./arsl_fcos_r50_fpn_coco_semi010.yml ) |
| ARSL-FCOS     |   10% |  **38.5(LSJ)**  | 240 (174240)  | [download](https://paddledet.bj.bcebos.com/models/arsl_fcos_r50_fpn_coco_semi010_lsj.pdparams) | [config](./arsl_fcos_r50_fpn_coco_semi010_lsj.yml ) |
| ARSL-FCOS     |   full(100%) |  **45.1**  | 240 (174240)  | [download](https://paddledet.bj.bcebos.com/models/arsl_fcos_r50_fpn_coco_full.pdparams) | [config](./arsl_fcos_r50_fpn_coco_full.yml ) |



## 使用说明

仅训练时必须使用半监督检测的配置文件去训练，评估、预测、部署也可以按基础检测器的配置文件去执行。

### 训练

```bash
# 单卡训练 (不推荐，需按线性比例相应地调整学习率)
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/semi_det/arsl/arsl_fcos_r50_fpn_coco_semi010.yml --eval

# 多卡训练
python -m paddle.distributed.launch --log_dir=arsl_fcos_r50_fpn_coco_semi010/ --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/semi_det/arsl/arsl_fcos_r50_fpn_coco_semi010.yml --eval
```

### 评估

```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/semi_det/arsl/arsl_fcos_r50_fpn_coco_semi010.yml -o weights=output/arsl_fcos_r50_fpn_coco_semi010/model_final.pdparams
```

### 预测

```bash
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/semi_det/arsl/arsl_fcos_r50_fpn_coco_semi010.yml -o weights=output/arsl_fcos_r50_fpn_coco_semi010/model_final.pdparams --infer_img=demo/000000014439.jpg
```


## 引用

```

```
