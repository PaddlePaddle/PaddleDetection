# 通用检测benchmark测试脚本说明

```
├── benchmark
│   ├── analysis_log.py
│   ├── prepare.sh
│   ├── README.md
│   ├── run_all.sh
│   ├── run_benchmark.sh
```

## 脚本说明

### prepare.sh
相关数据准备脚本，完成数据、模型的自动下载
### run_all.sh
主要运行脚本，可完成所有相关模型的测试方案
### run_benchmark.sh
单模型运行脚本，可完成指定模型的测试方案

## Docker 运行环境
* docker image: registry.baidubce.com/paddlepaddle/paddle:2.1.2-gpu-cuda10.2-cudnn7
* paddle = 2.1.2
* python = 3.7

## 运行benchmark测试

### 运行所有模型
```
git clone https://github.com/PaddlePaddle/PaddleDetection.git
cd PaddleDetection
bash benchmark/run_all.sh
```

### 运行指定模型
* Usage：bash run_benchmark.sh ${run_mode} ${batch_size} ${fp_item} ${max_epoch} ${model_name}
* model_name: faster_rcnn, fcos, deformable_detr, gfl, hrnet, higherhrnet, solov2, jde, fairmot
```
git clone https://github.com/PaddlePaddle/PaddleDetection.git
cd PaddleDetection
bash benchmark/prepare.sh

# 单卡
CUDA_VISIBLE_DEVICES=0 bash benchmark/run_benchmark.sh sp 2 fp32 1 faster_rcnn
# 多卡
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash benchmark/run_benchmark.sh mp 2 fp32 1 faster_rcnn
```
