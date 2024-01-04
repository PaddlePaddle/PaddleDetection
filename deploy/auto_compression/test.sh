#!/bin/bash

# 本脚本用于测试PaddleDection系列模型的自动压缩功能
## 运行脚本前，请确保处于以下环境：
## CUDA11.2+TensorRT8.0.3.4+Paddle2.5.2

## rtdetr_hgnetv2_l_6x_coco
## 启动自动化压缩训练
CUDA_VISIBLE_DEVICES=0 python run.py --save_dir ./models/rtdetr_hgnetv2_l_6x_coco_qat/ --act_config_path ./configs/rtdetr_hgnetv2_l_qat_dis.yaml --config_path ./configs/rtdetr_reader.yml
## GPU指标测试
### 量化前，预期指标：mAP:53.09%;time:32.7ms
python test_det.py --model_path ./models/rtdetr_hgnetv2_l_6x_coco/ --config ./configs/rtdetr_reader.yml --precision fp32 --use_trt True --use_dynamic_shape False
### 量化后，预期指标：mAP:52.92%;time:24.8ms
python test_det.py --model_path ./models/rtdetr_hgnetv2_l_6x_coco_qat/ --config ./configs/rtdetr_reader.yml --precision int8 --use_trt True --use_dynamic_shape False
## CPU指标测试
### 量化前，预期指标：mAP:52.54%;time:3392.0ms
python test_det.py --model_path ./models/rtdetr_hgnetv2_l_6x_coco/ --config ./configs/rtdetr_reader.yml --precision fp32 --use_mkldnn True --device CPU --cpu_threads 12 --use_dynamic_shape False
### 量化后，预期指标：mAP:52.95%;time:966.2ms
python test_det.py --model_path ./models/rtdetr_hgnetv2_l_6x_coco_qat/ --config ./configs/rtdetr_reader.yml --precision int8 --use_mkldnn True --device CPU --cpu_threads 12 --use_dynamic_shape False

## picodet_s_320_coco_lcnet
## 启动自动化压缩训练
CUDA_VISIBLE_DEVICES=0 python run.py --save_dir ./models/picodet_s_320_coco_lcnet_qat/ --act_config_path ./configs/picodet_s_320_lcnet_qat_dis.yaml --config_path ./configs/picodet_320_reader.yml
## GPU指标测试
### 量化前，预期指标：mAP:29.06%;time:3.6ms
python test_det.py --model_path ./models/picodet_s_320_coco_lcnet/ --config ./configs/picodet_320_reader.yml --precision fp32 --use_trt True --use_dynamic_shape False
### 量化后，预期指标：mAP:28.82%;time:3.3ms
python test_det.py --model_path ./models/picodet_s_320_coco_lcnet_qat/ --config ./configs/picodet_320_reader.yml --precision int8 --use_trt True --use_dynamic_shape False
## CPU指标测试
### 量化前，预期指标：mAP:29.06%;time:42.0ms
python test_det.py --model_path ./models/picodet_s_320_coco_lcnet/ --config ./configs/picodet_320_reader.yml --precision fp32 --use_mkldnn True --device CPU --cpu_threads 12 --use_dynamic_shape False
### 量化后，预期指标：mAP:28.58%;time:46.7ms
python test_det.py --model_path ./models/picodet_s_320_coco_lcnet_qat/ --config ./configs/picodet_320_reader.yml --precision int8 --use_mkldnn True --device CPU --cpu_threads 12 --use_dynamic_shape False

## ppyoloe_plus_crn_l_80e_coco
## 启动自动化压缩训练
CUDA_VISIBLE_DEVICES=0 python run.py --save_dir ./models/ppyoloe_plus_crn_l_80e_coco_qat/ --act_config_path ./configs/ppyoloe_plus_l_qat_dis.yaml --config_path ./configs/ppyoloe_plus_reader.yml
## GPU指标测试
### 量化前，预期指标：mAP:52.88%;time:12.4ms
python test_det.py --model_path ./models/ppyoloe_plus_crn_l_80e_coco/ --config ./configs/ppyoloe_plus_reader.yml --precision fp32 --use_trt True --use_dynamic_shape False
### 量化后，预期指标：mAP:52.52%;time:7.2ms
python test_det.py --model_path ./models/ppyoloe_plus_crn_l_80e_coco_qat/ --config ./configs/ppyoloe_plus_reader.yml --precision int8 --use_trt True --use_dynamic_shape False
## CPU指标测试
### 量化前，预期指标：mAP:52.88%;time:522.6ms
python test_det.py --model_path ./models/ppyoloe_plus_crn_l_80e_coco/ --config ./configs/ppyoloe_plus_reader.yml --precision fp32 --use_mkldnn True --device CPU --cpu_threads 12 --use_dynamic_shape False
### 量化后，预期指标：mAP:52.65%;time:539.5ms
python test_det.py --model_path ./models/ppyoloe_plus_crn_l_80e_coco_qat/ --config ./configs/ppyoloe_plus_reader.yml --precision int8 --use_mkldnn True --device CPU --cpu_threads 12 --use_dynamic_shape False

## dino_r50_4scale_2x
## 启动自动化压缩训练
CUDA_VISIBLE_DEVICES=0 python run.py --save_dir ./models/dino_r50_4scale_2x_qat/ --act_config_path configs/dino_r50_4scale_2x_qat_dis.yaml --config_path ./configs/dino_reader.yml
## GPU指标测试
### 量化前，预期指标：mAP:53.09%;time:147.7ms
python test_det.py --model_path ./models/dino_r50_4scale_2x/ --config ./configs/dino_reader.yml --precision fp32 --use_trt True
### 量化后，预期指标：mAP:52.92%;time:127.9ms
python test_det.py --model_path ./models/dino_r50_4scale_2x_qat/ --config ./configs/dino_reader.yml --precision int8 --use_trt True
## CPU指标测试需在develop版本的Paddle下才能正常跑通，测试命令如下：
### 量化前
# python test_det.py --model_path ./models/rtdetr_hgnetv2_l_6x_coco/ --config ./configs/rtdetr_reader.yml --precision fp32 --use_mkldnn True --device CPU --cpu_threads 12
### 量化后
# python test_det.py --model_path ./models/rtdetr_hgnetv2_l_6x_coco_qat/ --config ./configs/rtdetr_reader.yml --precision int8 --use_mkldnn True --device CPU --cpu_threads 12
