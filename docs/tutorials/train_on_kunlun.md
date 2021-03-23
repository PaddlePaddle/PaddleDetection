# How to train on kunlun

## Prepare kunlun environment
[Paddle installation for machines with Kunlun XPU card](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/2.0-rc1/install/install_Kunlun_zh.html)

## yolov3

### Prepare data
Download dataset from https://paddlemodels.bj.bcebos.com/object_detection/roadsign_voc.tar, and put it in teh dataset directory.


### Train
```shell
python3.7 -u tools/train.py -c configs/yolov3_mobilenet_v1_roadsign.yml -o use_gpu=False use_xpu=True
```


### Eval
```shell
python3.7 -u tools/eval.py -c configs/yolov3_mobilenet_v1_roadsign.yml -o weights=output/yolov3_mobilenet_v1_roadsign/model_final.pdparams use_gpu=False use_xpu=True
```


## ppyolo

### Prepare data
Prepare data roadsign


### Train
```shell
python3.7 -u tools/train.py --eval -c configs/ppyolo/ppyolo_roadsign_kunlun.yml
```


### Eval
```shell
python3.7 -u tools/eval.py -c configs/ppyolo/ppyolo_roadsign_kunlun.yml
```


## mask_rcnn

### Prepare data
Download dataset from https://dataset.bj.bcebos.com/PaddleDetection_demo/cocome.tar and put it in the dataset directory.



### Train
```shell
python3.7 -u tools/train.py --eval -c configs/mask_rcnn_r50_1x_cocome_kunlun.yml
```


### Eval
```shell
python3.7 -u tools/eval.py -c configs/mask_rcnn_r50_1x_cocome_kunlun.yml
```
