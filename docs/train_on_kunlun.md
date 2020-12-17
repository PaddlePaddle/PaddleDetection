# How to train on kunlun

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
python3.7 -u tools/train.py -c configs/ppyolo/ppyolo_roadsign_kunlun.yml
```


### eval
```shell
python3.7 -u tools/eval.py -c configs/ppyolo/ppyolo_roadsign_kunlun.yml
```
