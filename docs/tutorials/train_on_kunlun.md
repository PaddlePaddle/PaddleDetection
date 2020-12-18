# How to train on kunlun

## prepare kunlun environment
[Paddle installation for machines with Kunlun XPU card](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/2.0-rc1/install/install_Kunlun_zh.html)

## yolov3

### Prepare data
Prepare data roadsign


### train
```shell
python3.7 -u tools/train.py -c configs/yolov3_mobilenet_v1_roadsign.yml -o use_gpu=False use_xpu=True
```


### eval
```shell
python3.7 -u tools/eval.py -c configs/yolov3_mobilenet_v1_roadsign.yml -o weights=output/yolov3_mobilenet_v1_roadsign/model_final.pdparams use_gpu=False use_xpu=True
```


## ppyolo

### Prepare data
Prepare data roadsign


### train
```shell
python3.7 -u tools/train.py --eval -c configs/ppyolo/ppyolo_roadsign_kunlun.yml
```


### eval
```shell
python3.7 -u tools/eval.py -c configs/ppyolo/ppyolo_roadsign_kunlun.yml
```
