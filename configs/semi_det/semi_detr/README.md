简体中文 | [English](README_en.md)

# RTDETR-SSOD(基于RTDETR的配套半监督目标检测方法)

#复现模型指标注意事项: 模型中指标是采用在监督数据训练饱和后加载监督数据所训练的模型进行半监督训练
    例如 使用 baseline/rtdetr_r50vd_6x_coco_sup005.yml使用5%coco数据训练全监督模型，得到rtdetr_r50vd_6x_coco_sup005.pdparams,在rt_detr_ssod005_coco_no_warmup.yml中设置
            pretrain_student_weights: rtdetr_r50vd_6x_coco_sup005.pdparams 
            pretrain_teacher_weights: rtdetr_r50vd_6x_coco_sup005.pdparams
    1.使用coco数据集5%和10%有标记数据和voc数据集VOC2007trainval 所训练的权重已给出请参考 semi_det/baseline/README.md.
    2.rt_detr_ssod_voc_no_warmup.yml rt_detr_ssod005_coco_no_warmup.yml rt_detr_ssod010_coco_no_warmup.yml 是使用训练好的全监督权中直接开启半监督训练
## RTDETR-SSOD模型库

|      模型       |  监督数据比例 |        Sup Baseline     |    Sup Epochs (Iters)   |  Sup mAP<sup>val<br>0.5:0.95 | Semi mAP<sup>val<br>0.5:0.95 |  Semi Epochs (Iters)  |  模型下载  |   配置文件   |
| :------------: | :---------: | :---------------------: | :---------------------: |:---------------------------: |:----------------------------: | :------------------: |:--------: |:----------: |
| RTDETR-SSOD    |   5% |   [sup_config](../baseline/rtdetr_r50vd_6x_coco_sup005.yml)    |  - | 39.0 |  **42.3**  | -  | [download](https://paddledet.bj.bcebos.com/models/rt_detr_ssod005_coco_no_warmup.pdparams) | [config](./rt_detr_ssod005_coco_no_warmup.yml) |
| RTDETR-SSOD     |   10% |   [sup_config](../baseline/rtdetr_r50vd_6x_coco_sup010.yml)    | -| 42.3 |  **44.8**  | - | [download](https://paddledet.bj.bcebos.com/models/rt_detr_ssod010_coco_no_warmup.pdparams) | [config](./rt_detr_ssod010_coco_with_warmup.yml) |
| RTDETR-SSOD(VOC)|   10% |   [sup_config](../baseline/rtdetr_r50vd_6x_coco_voc2007.yml)    | -  | 62.7 |  **65.8(LSJ)**  | -  | [download](https://paddledet.bj.bcebos.com/models/rt_detr_ssod_voc_no_warmup.pdparams) | [config](./rt_detr_ssod_voc_with_warmup.yml) |

**注意:**
 - 以上模型训练默认使用8 GPUs，监督数据总batch_size默认为16，无监督数据总batch_size默认也为16，默认初始学习率为0.01。如果改动了总batch_size，请按线性比例相应地调整学习率；
 - **监督数据比例**是指使用的有标签COCO数据集占 COCO train2017 全量训练集的百分比，使用的无标签COCO数据集一般也是相同比例，但具体图片和有标签数据的图片不重合；
 - `Semi Epochs (Iters)`表示**半监督训练**的模型的 Epochs (Iters)，如果使用**自定义数据集**，需自行根据Iters换算到对应的Epochs调整，最好保证总Iters 和COCO数据集的设置较为接近；
 - `Sup mAP`是**只使用有监督数据训练**的模型的精度，请参照**基础检测器的配置文件** 和 [baseline](../baseline)；
 - `Semi mAP`是**半监督训练**的模型的精度，模型下载和配置文件的链接均为**半监督模型**；
 - `LSJ`表示 **large-scale jittering**，表示使用更大范围的多尺度训练，可进一步提升精度，但训练速度也会变慢；
 - 半监督检测的配置讲解，请参照[文档](../README.md/#半监督检测配置)；
 - `Dense Teacher`原文使用`R50-va-caffe`预训练，PaddleDetection中默认使用`R50-vb`预训练，如果使用`R50-vd`结合[SSLD](../../../docs/feature_models/SSLD_PRETRAINED_MODEL.md)的预训练模型，可进一步显著提升检测精度，同时backbone部分配置也需要做出相应更改，如：
 ```python
  pretrain_weights:  https://paddledet.bj.bcebos.com/models/pretrained/ResNet50_vd_ssld_v2_pretrained.pdparams
  ResNet:
    depth: 50
    variant: d
    norm_type: bn
    freeze_at: 0
    return_idx: [1, 2, 3]
    num_stages: 4
    lr_mult_list: [0.05, 0.05, 0.1, 0.15]
 ```

## 使用说明

仅训练时必须使用半监督检测的配置文件去训练，评估、预测、部署也可以按基础检测器的配置文件去执行。

### 训练

```bash
# 单卡训练 (不推荐，需按线性比例相应地调整学习率)
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/semi_det/denseteacher/denseteacher_fcos_r50_fpn_coco_semi010.yml --eval

# 多卡训练
python -m paddle.distributed.launch --log_dir=denseteacher_fcos_semi010/ --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/semi_det/denseteacher/denseteacher_fcos_r50_fpn_coco_semi010.yml --eval
```

### 评估

```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/semi_det/denseteacher/denseteacher_fcos_r50_fpn_coco_semi010.yml -o weights=output/denseteacher_fcos_r50_fpn_coco_semi010/model_final.pdparams
```

### 预测

```bash
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/semi_det/denseteacher/denseteacher_fcos_r50_fpn_coco_semi010.yml -o weights=output/denseteacher_fcos_r50_fpn_coco_semi010/model_final.pdparams --infer_img=demo/000000014439.jpg
```

### 部署

部署可以使用半监督检测配置文件，也可以使用基础检测器的配置文件去部署和使用。

```bash
# 导出模型
CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c configs/semi_det/denseteacher/denseteacher_fcos_r50_fpn_coco_semi010.yml -o weights=https://paddledet.bj.bcebos.com/models/denseteacher_fcos_r50_fpn_coco_semi010.pdparams

# 导出权重预测
CUDA_VISIBLE_DEVICES=0 python deploy/python/infer.py --model_dir=output_inference/denseteacher_fcos_r50_fpn_coco_semi010 --image_file=demo/000000014439_640x640.jpg --device=GPU

# 部署测速
CUDA_VISIBLE_DEVICES=0 python deploy/python/infer.py --model_dir=output_inference/denseteacher_fcos_r50_fpn_coco_semi010 --image_file=demo/000000014439_640x640.jpg --device=GPU --run_benchmark=True # --run_mode=trt_fp16

# 导出ONNX
paddle2onnx --model_dir output_inference/denseteacher_fcos_r50_fpn_coco_semi010/ --model_filename model.pdmodel --params_filename model.pdiparams --opset_version 12 --save_file denseteacher_fcos_r50_fpn_coco_semi010.onnx
```


## 引用

```
 @article{denseteacher2022,
  title={Dense Teacher: Dense Pseudo-Labels for Semi-supervised Object Detection},
  author={Hongyu Zhou, Zheng Ge, Songtao Liu, Weixin Mao, Zeming Li, Haiyan Yu, Jian Sun},
  journal={arXiv preprint arXiv:2207.02541},
  year={2022}
}
```
