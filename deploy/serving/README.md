# 服务端预测部署

`PaddleDetection`训练出来的模型可以使用[Serving](https://github.com/PaddlePaddle/Serving) 部署在服务端。  
本教程以在COCO数据集上用`configs/yolov3/yolov3_darknet53_270e_coco.yml`算法训练的模型进行部署。  
预训练模型权重文件为[yolov3_darknet53_270e_coco.pdparams](https://paddledet.bj.bcebos.com/models/yolov3_darknet53_270e_coco.pdparams) 。

## 1. 首先验证模型
```
python tools/infer.py -c configs/yolov3/yolov3_darknet53_270e_coco.yml --infer_img=demo/000000014439.jpg -o use_gpu=True weights=https://paddledet.bj.bcebos.com/models/yolov3_darknet53_270e_coco.pdparams --infer_img=demo/000000014439.jpg
```

## 2. 安装 paddle serving
请参考[PaddleServing](https://github.com/PaddlePaddle/Serving/tree/v0.7.0) 中安装教程安装（版本>=0.7.0）。

## 3. 导出模型
PaddleDetection在训练过程包括网络的前向和优化器相关参数，而在部署过程中，我们只需要前向参数，具体参考:[导出模型](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.6/deploy/EXPORT_MODEL.md)

```
python tools/export_model.py -c configs/yolov3/yolov3_darknet53_270e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/yolov3_darknet53_270e_coco.pdparams --export_serving_model=True
```

以上命令会在`output_inference/`文件夹下生成一个`yolov3_darknet53_270e_coco`文件夹：
```
output_inference
│   ├── yolov3_darknet53_270e_coco
│   │   ├── infer_cfg.yml
│   │   ├── model.pdiparams
│   │   ├── model.pdiparams.info
│   │   ├── model.pdmodel
│   │   ├── serving_client
│   │   │   ├── serving_client_conf.prototxt
│   │   │   ├── serving_client_conf.stream.prototxt
│   │   ├── serving_server
│   │   │   ├── __model__
│   │   │   ├── __params__
│   │   │   ├── serving_server_conf.prototxt
│   │   │   ├── serving_server_conf.stream.prototxt
│   │   │   ├── ...
```

`serving_client`文件夹下`serving_client_conf.prototxt`详细说明了模型输入输出信息
`serving_client_conf.prototxt`文件内容为：
```
feed_var {
  name: "im_shape"
  alias_name: "im_shape"
  is_lod_tensor: false
  feed_type: 1
  shape: 2
}
feed_var {
  name: "image"
  alias_name: "image"
  is_lod_tensor: false
  feed_type: 1
  shape: 3
  shape: 608
  shape: 608
}
feed_var {
  name: "scale_factor"
  alias_name: "scale_factor"
  is_lod_tensor: false
  feed_type: 1
  shape: 2
}
fetch_var {
  name: "multiclass_nms3_0.tmp_0"
  alias_name: "multiclass_nms3_0.tmp_0"
  is_lod_tensor: true
  fetch_type: 1
  shape: -1
}
fetch_var {
  name: "multiclass_nms3_0.tmp_2"
  alias_name: "multiclass_nms3_0.tmp_2"
  is_lod_tensor: false
  fetch_type: 2
```

## 4. 启动PaddleServing服务

```
cd output_inference/yolov3_darknet53_270e_coco/

# GPU
python -m paddle_serving_server.serve --model serving_server --port 9393 --gpu_ids 0

# CPU
python -m paddle_serving_server.serve --model serving_server --port 9393
```

## 5. 测试部署的服务
准备`label_list.txt`文件，示例`label_list.txt`文件内容为
```
person
bicycle
car
motorcycle
airplane
bus
train
truck
boat
traffic light
fire hydrant
stop sign
parking meter
bench
bird
cat
dog
horse
sheep
cow
elephant
bear
zebra
giraffe
backpack
umbrella
handbag
tie
suitcase
frisbee
skis
snowboard
sports ball
kite
baseball bat
baseball glove
skateboard
surfboard
tennis racket
bottle
wine glass
cup
fork
knife
spoon
bowl
banana
apple
sandwich
orange
broccoli
carrot
hot dog
pizza
donut
cake
chair
couch
potted plant
bed
dining table
toilet
tv
laptop
mouse
remote
keyboard
cell phone
microwave
oven
toaster
sink
refrigerator
book
clock
vase
scissors
teddy bear
hair drier
toothbrush
```

设置`prototxt`文件路径为`serving_client/serving_client_conf.prototxt`
设置`fetch`为`fetch=["multiclass_nms3_0.tmp_0"])`

测试
```
# 进入目录
cd output_inference/yolov3_darknet53_270e_coco/

# 测试代码 test_client.py 会自动创建output文件夹，并在output下生成`bbox.json`和`000000014439.jpg`两个文件
python ../../deploy/serving/test_client.py ../../deploy/serving/label_list.txt ../../demo/000000014439.jpg
```
