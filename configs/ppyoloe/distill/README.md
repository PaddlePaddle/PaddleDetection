# PPYOLOE+ Distillation(PPYOLOE+ 蒸馏)

PaddleDetection提供了对PPYOLOE+ 进行模型蒸馏的方案，结合了logits蒸馏和feature蒸馏。更多蒸馏方案可以查看[slim/distill](../../slim/distill/)。

## 模型库

| 模型               |    方案     | 输入尺寸 | epochs |    Box mAP    |       配置文件    |     下载链接    |
| ----------------- | ----------- | ------ | :----: | :-----------: | :--------------: | :------------: |
|   PP-YOLOE+_x     |  teacher   |  640     | 80e   |      54.7     | [config](../ppyoloe_plus_crn_x_80e_coco.yml) | [model](https://bj.bcebos.com/v1/paddledet/models/ppyoloe_plus_crn_x_80e_coco.pdparams) |
|   PP-YOLOE+_l     |  student   |  640     | 80e   |      52.9     | [config](../ppyoloe_plus_crn_l_80e_coco.yml) | [model](https://bj.bcebos.com/v1/paddledet/models/ppyoloe_plus_crn_l_80e_coco.pdparams) |
|   PP-YOLOE+_l     |  distill   |  640     | 80e   |   **54.0(+1.1)**  | [config](./ppyoloe_plus_crn_l_80e_coco_distill.yml),[slim_config](../../slim/distill/ppyoloe_plus_distill_x_distill_l.yml)  | [model](https://bj.bcebos.com/v1/paddledet/models/ppyoloe_plus_crn_l_80e_coco_distill.pdparams) |
|   PP-YOLOE+_l     |  teacher   |  640     | 80e   |      52.9     | [config](../ppyoloe_plus_crn_l_80e_coco.yml) | [model](https://bj.bcebos.com/v1/paddledet/models/ppyoloe_plus_crn_l_80e_coco.pdparams) |
|   PP-YOLOE+_m     |  student   |  640     | 80e   |      49.8     | [config](../ppyoloe_plus_crn_m_80e_coco.yml) | [model](https://bj.bcebos.com/v1/paddledet/models/ppyoloe_plus_crn_m_80e_coco.pdparams) |
|   PP-YOLOE+_m     |  distill   |  640     | 80e   |    **51.0(+1.2)**    | [config](./ppyoloe_plus_crn_m_80e_coco_distill.yml),[slim_config](../../slim/distill/ppyoloe_plus_distill_l_distill_m.yml)  | [model](https://bj.bcebos.com/v1/paddledet/models/ppyoloe_plus_crn_m_80e_coco_distill.pdparams) |

## 快速开始

### 训练
```shell
# 单卡
python tools/train.py -c configs/ppyoloe/distill/ppyoloe_plus_crn_l_80e_coco_distill.yml --slim_config configs/slim/distill/ppyoloe_plus_distill_x_distill_l.yml
# 多卡
python -m paddle.distributed.launch --log_dir=ppyoloe_plus_distill_x_distill_l/ --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/ppyoloe/distill/ppyoloe_plus_crn_l_80e_coco_distill.yml --slim_config configs/slim/distill/ppyoloe_plus_distill_x_distill_l.yml
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
