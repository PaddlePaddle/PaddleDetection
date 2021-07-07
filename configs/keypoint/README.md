# KeyPoint模型系列



## 简介

-    PaddleDetection KeyPoint部分紧跟业内最新最优算法方案，包含Top-Down、BottomUp两套方案，以满足用户的不同需求。

<div align="center">
  <img src="./football_keypoint.gif" width='800'/>
</div>



####   Model Zoo

| 模型              | 输入尺寸 | 通道数 | AP(coco val) |                           模型下载                           | 配置文件                                                    |
| :---------------- | -------- | ------ | :----------: | :----------------------------------------------------------: | ----------------------------------------------------------- |
| HigherHRNet       | 512      | 32     |     67.1     | [higherhrnet_hrnet_w32_512.pdparams](https://paddledet.bj.bcebos.com/models/keypoint/higherhrnet_hrnet_w32_512.pdparams) | [config](./higherhrnet/higherhrnet_hrnet_w32_512.yml)       |
| HigherHRNet       | 640      | 32     |     68.3     | [higherhrnet_hrnet_w32_640.pdparams](https://paddledet.bj.bcebos.com/models/keypoint/higherhrnet_hrnet_w32_640.pdparams) | [config](./higherhrnet/higherhrnet_hrnet_w32_640.yml)       |
| HigherHRNet+SWAHR | 512      | 32     |     68.9     | [higherhrnet_hrnet_w32_512_swahr.pdparams](https://paddledet.bj.bcebos.com/models/keypoint/higherhrnet_hrnet_w32_512_swahr.pdparams) | [config](./higherhrnet/higherhrnet_hrnet_w32_512_swahr.yml) |
| HRNet             | 256x192  | 32     |     76.9     | [hrnet_w32_256x192.pdparams](https://paddledet.bj.bcebos.com/models/keypoint/hrnet_w32_256x192.pdparams) | [config](./hrnet/hrnet_w32_256x192.yml)                     |
| HRNet             | 384x288  | 32     |     77.8     | [hrnet_w32_384x288.pdparams](https://paddledet.bj.bcebos.com/models/keypoint/hrnet_w32_384x288.pdparams) | [config](./hrnet/hrnet_w32_384x288.yml)                     |



## 快速开始

### 1、环境安装

​    请参考PaddleDetection [安装文档](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/docs/tutorials/INSTALL_cn.md)正确安装PaddlePaddle和PaddleDetection即可

### 2、数据准备

​    目前KeyPoint模型基于coco数据集开发，其他数据集尚未验证

​    请参考PaddleDetection[数据准备部分](https://github.com/PaddlePaddle/PaddleDetection/blob/f0a30f3ba6095ebfdc8fffb6d02766406afc438a/docs/tutorials/PrepareDataSet.md)部署准备COCO数据集即可

### 3、训练与测试

​    **单卡训练：**

```shell
CUDA_VISIBLE_DEVICES=0 python3 tools/train.py -c configs/keypoint/higherhrnet/higherhrnet_hrnet_w32_512.yml
```

​    **多卡训练：**

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m paddle.distributed.launch tools/train.py -c configs/keypoint/higherhrnet/higherhrnet_hrnet_w32_512.yml
```

​    **模型评估：**

```shell
CUDA_VISIBLE_DEVICES=0 python3 tools/eval.py -c configs/keypoint/higherhrnet/higherhrnet_hrnet_w32_512.yml
```

​    **模型预测：**

​    注意：top-down模型只支持单人截图预测，如需使用多人图，请使用[联合部署推理]方式。或者使用bottom-up模型。

```shell
CUDA_VISIBLE_DEVICES=0 python3 tools/infer.py -c configs/keypoint/higherhrnet/higherhrnet_hrnet_w32_512.yml -o weights=./output/higherhrnet_hrnet_w32_512/model_final.pdparams --infer_dir=../images/ --draw_threshold=0.5 --save_txt=True
```

​    **部署预测：**

```shell
#导出模型
python tools/export_model.py -c configs/keypoint/higherhrnet/higherhrnet_hrnet_w32_512.yml -o weights=output/higherhrnet_hrnet_w32_512/model_final.pdparams

#部署推理
#keypoint top-down/bottom-up 单独推理，该模式下top-down模型只支持单人截图预测。
python deploy/python/keypoint_infer.py --model_dir=output_inference/higherhrnet_hrnet_w32_512/ --image_file=./demo/000000014439_640x640.jpg --use_gpu=True --threshold=0.5
python deploy/python/keypoint_infer.py --model_dir=output_inference/hrnet_w32_384x288/ --image_file=./demo/hrnet_demo.jpg --use_gpu=True --threshold=0.5

#keypoint top-down模型 + detector 检测联合部署推理（联合推理只支持top-down方式）
python deploy/python/keypoint_det_unite_infer.py --det_model_dir=output_inference/ppyolo_r50vd_dcn_2x_coco/ --keypoint_model_dir=output_inference/hrnet_w32_384x288/ --video_file=../video/xxx.mp4  --use_gpu=True
```
