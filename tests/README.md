# 全链条CI/CE脚本详细说明

```
├── tests
│   ├── ppdet_params
│   │   ├── ppyolo_mbv3_large_coco_params.txt
│   │   ├── ppyolo_r50vd_dcn_1x_coco_params.txt
│   │   ├── ppyolov2_r50vd_dcn_365e_coco_params.txt
│   │   ├── yolov3_darknet53_270e_coco_params.txt
│   │   ├── ...
│   ├── prepare.sh
│   ├── README.md
│   ├── requirements.txt
│   ├── test.sh
```

## 脚本说明

### test.sh
主要运行脚本，以 `ppdet_params/xxx_params.txt` 作为参数配置，可完成相关测试方案
### prepare.sh
相关数据准备脚本，以 `ppdet_params/xxx_params.txt` 作为参数配置，完成数据、预训练模型、预测模型的自动下载

## test.sh使用方法
以 `yolov3_darknet53_270e_coco_params.txt` 为例，进入到PaddleDetection目录下

### 模式1： 少量数据训练，少量数据预测（lite_train_infer）
```bash
bash ./tests/prepare.sh ./tests/ppdet_params/yolov3_darknet53_270e_coco_params.txt lite_train_infer
bash ./tests/test.sh ./tests/ppdet_params/yolov3_darknet53_270e_coco_params.txt lite_train_infer
```

### 模式2： 少量数据训练，全量数据预测（whole_infer）
```bash
bash ./tests/prepare.sh ./tests/ppdet_params/yolov3_darknet53_270e_coco_params.txt whole_infer
bash ./tests/test.sh ./tests/ppdet_params/yolov3_darknet53_270e_coco_params.txt whole_infer
```

### 模式3： 全量数据训练，全量数据预测（whole_train_infer）
```bash
bash ./tests/prepare.sh ./tests/ppdet_params/yolov3_darknet53_270e_coco_params.txt whole_train_infer
bash ./tests/test.sh ./tests/ppdet_params/yolov3_darknet53_270e_coco_params.txt whole_train_infer
```

### 模式4： 不训练，全量数据预测（infer）
```bash
bash ./tests/prepare.sh ./tests/ppdet_params/yolov3_darknet53_270e_coco_params.txt infer
bash ./tests/test.sh ./tests/ppdet_params/yolov3_darknet53_270e_coco_params.txt infer
```
**注：**
运行`prepare.sh`时，会清空`dataset/coco/`下所有文件
