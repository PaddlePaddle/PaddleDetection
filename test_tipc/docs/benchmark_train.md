# TIPC Linux端Benchmark测试文档

该文档为Benchmark测试说明，Benchmark预测功能测试的主程序为`benchmark_train.sh`，用于验证监控模型训练的性能。

# 1. 测试流程
## 1.1 准备数据和环境安装
运行`test_tipc/prepare.sh`，完成训练数据准备和安装环境流程。

```shell
# 运行格式：bash test_tipc/prepare.sh  train_benchmark.txt  mode
bash test_tipc/prepare.sh test_tipc/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco_train_infer_python.txt benchmark_train
```

## 1.2 功能测试
执行`test_tipc/benchmark_train.sh`，完成模型训练和日志解析

```shell
# 运行格式：bash test_tipc/benchmark_train.sh train_benchmark.txt mode
bash test_tipc/benchmark_train.sh test_tipc/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco_train_infer_python.txt benchmark_train
```

`test_tipc/benchmark_train.sh`支持根据传入的第三个参数实现只运行某一个训练配置，如下：
```shell
# 运行格式：bash test_tipc/benchmark_train.sh train_benchmark.txt mode
bash test_tipc/benchmark_train.sh test_tipc/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco_train_infer_python.txt benchmark_train  dynamic_bs2_fp32_DP_N1C1
```
dynamic_bs2_fp32_DP_N1C1为test_tipc/benchmark_train.sh传入的参数，格式如下：
`${modeltype}_${batch_size}_${fp_item}_${run_mode}_${device_num}`
包含的信息有：模型类型、batchsize大小、训练精度如fp32,fp16等、分布式运行模式以及分布式训练使用的机器信息如单机单卡（N1C1）。


## 2. 日志输出

运行后将保存模型的训练日志和解析日志，使用 `test_tipc/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco_train_infer_python.txt` 参数文件的训练日志解析结果是：

```
{"model_branch": "tipc_fuse_benchmark", "model_commit": "4cce901d231f7954468045cf96302505bd6be495", "model_name": "faster_rcnn_r50_fpn_1x_coco_bs2_fp32_SingleP_DP", "batch_size": 2, "fp_item": "fp32", "run_process_type": "SingleP", "run_mode": "DP", "convergence_value": "0.556966", "convergence_key": "loss:", "ips": 4.857, "speed_unit": "images/s", "device_num": "N1C1", "model_run_time": "590", "frame_commit": "6b0c57cf65945e97d87a8fba89c0a2fc18dd8544", "frame_version": "0.0.0"}
```

训练日志和日志解析结果保存在benchmark_log目录下，文件组织格式如下：
```
train_log/
├── index
│   └── PaddleDetection_faster_rcnn_r50_fpn_1x_coco_bs2_fp32_SingleP_DP_N1C1_speed
├── profiling_log
│   └── PaddleDetection_faster_rcnn_r50_fpn_1x_coco_bs2_fp32_SingleP_DP_N1C1_profiling
└── train_log
    ├── PaddleDetection_faster_rcnn_r50_fpn_1x_coco_bs2_fp32_SingleP_DP_N1C1_log
    └── PaddleDetection_faster_rcnn_r50_fpn_1x_coco_bs2_fp32_MultiP_DP_N1C4_log
```
