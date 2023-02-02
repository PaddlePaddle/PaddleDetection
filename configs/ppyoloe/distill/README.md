# PPYOLOE+ Distillation(PPYOLOE+ 蒸馏)

PaddleDetection提供了对PPYOLOE+ 进行模型蒸馏的方案，结合了logits蒸馏和feature蒸馏。


## 模型库



## 快速开始

### 训练
```shell
# 单卡
python tools/train.py -c configs/ppyoloe/distill/ppyoloe_plus_crn_l_80e_coco_distill.yml --slim_config configs/slim/distill/ppyoloe_plus_distill_x_to_l.yml
# 多卡
python3.7 -m paddle.distributed.launch --log_dir=ppyoloe_plus_distill_x_to_l/ --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/ppyoloe/distill/ppyoloe_plus_crn_l_80e_coco_distill.yml --slim_config configs/slim/distill/ppyoloe_plus_distill_x_to_l.yml
```

- `-c`: 指定模型配置文件，也是student配置文件。
- `--slim_config`: 指定压缩策略配置文件，也是teacher配置文件。

### 评估
```shell
python tools/eval.py -c configs/ppyoloe/distill/ppyoloe_plus_crn_l_80e_coco_distill.yml -o weights=output/ppyoloe_plus_crn_l_80e_coco_distill/model_final.pdparams
```

- `-c`: 指定模型配置文件，也是student配置文件。
- `--slim_config`: 指定压缩策略配置文件，也是teacher配置文件。
- `-o weights`: 指定压缩算法训好的模型路径。

### 测试
```shell
python tools/infer.py -c configs/ppyoloe/distill/ppyoloe_plus_crn_l_80e_coco_distill.yml -o weights=output/ppyoloe_plus_crn_l_80e_coco_distill/model_final.pdparams --infer_img=demo/000000014439_640x640.jpg
```

- `-c`: 指定模型配置文件。
- `--slim_config`: 指定压缩策略配置文件。
- `-o weights`: 指定压缩算法训好的模型路径。
- `--infer_img`: 指定测试图像路径。
