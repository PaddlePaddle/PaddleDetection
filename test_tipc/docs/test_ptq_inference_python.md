# Linux GPU/CPU 离线量化功能测试

Linux GPU/CPU 离线量化功能测试的主程序为`test_ptq_inference_python.sh`，可以测试基于Python的离线量化功能。

## 1. 测试结论汇总

| 模型类型 |device | batchsize | tensorrt | mkldnn | cpu多线程 |
|  ----   |  ---- |-----------|  :----:  |   :----:   |  :----:  |
| 量化模型 | GPU | 1/2       | int8 | - | - |
| 量化模型 | CPU | 1/2       | - | int8 | 支持 |

## 2. 测试流程
### 2.1 功能测试
先运行`prepare.sh`准备数据和模型，然后运行`test_ptq_inference_python.sh`进行测试，最终在```test_tipc/output```目录下生成`python_infer_*.log`后缀的日志文件。

```shell
bash test_tipc/prepare.sh ./test_tipc/configs/yolov3/yolov3_darknet53_270e_coco_train_ptq_infer_python.txt "whole_infer"

# 用法:
bash test_tipc/test_ptq_inference_python.sh ./test_tipc/configs/yolov3/yolov3_darknet53_270e_coco_train_ptq_infer_python.txt
```  

#### 运行结果

各测试的运行情况会打印在 `test_tipc/output/results_ptq_python.log` 中：
运行成功时会输出：

```
Run successfully with command - yolov3_darknet53_270e_coco - python3.7 tools/post_quant.py -c configs/yolov3/yolov3_darknet53_270e_coco.yml --slim_config configs/slim/post_quant/yolov3_darknet53_ptq.yml -o weights=https://paddledet.bj.bcebos.com/models/yolov3_darknet53_270e_coco.pdparams filename=yolov3_darknet53_270e_coco --output_dir=./output_inference !
Run successfully with command - yolov3_darknet53_270e_coco - python3.7 ./deploy/python/infer.py --device=gpu --run_mode=paddle --model_dir=./output_inference/yolov3_darknet53_270e_coco --batch_size=2 --image_dir=./dataset/coco/test2017/ --run_benchmark=False   > ./test_tipc/output/yolov3_darknet53_270e_coco/whole_infer/python_infer_gpu_mode_paddle_batchsize_2.log 2>&1 !
...
```

运行失败时会输出：

```
Run failed with command - yolov3_darknet53_270e_coco - python3.7 tools/post_quant.py -c configs/yolov3/yolov3_darknet53_270e_coco.yml --slim_config configs/slim/post_quant/yolov3_darknet53_ptq.yml -o weights=https://paddledet.bj.bcebos.com/models/yolov3_darknet53_270e_coco.pdparams filename=yolov3_darknet53_270e_coco --output_dir=./output_inference!
...
```


## 3. 更多教程

本文档为功能测试用，更详细的离线量化功能使用教程请参考：[Paddle 离线量化官网教程](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/api_cn/static/quant/quantization_api.rst#quant_post_static)
