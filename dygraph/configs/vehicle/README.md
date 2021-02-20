English | [简体中文](CONTRIB_cn.md)
# PaddleDetection applied for specific scenarios

We provide some models implemented by PaddlePaddle to detect objects in specific scenarios, users can download the models and use them in these scenarios.

| Task                 | Algorithm | Box AP | Download                                                                                | Configs |
|:---------------------|:---------:|:------:| :-------------------------------------------------------------------------------------: |:------:|
| Vehicle Detection    |  YOLOv3  |  54.5  | [model](https://paddledet.bj.bcebos.com/models/vehicle_yolov3_darknet.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/vehicle/vehicle_yolov3_darknet.yml) |

## Vehicle Detection

One of major applications of vehichle detection is traffic monitoring. In this scenary, vehicles to be detected are mostly captured by the cameras mounted on top of traffic light columns.

### 1. Network

The network for detecting vehicles is YOLOv3, the backbone of which is Dacknet53.

### 2. Configuration for training

PaddleDetection provides users with a configuration file [yolov3_darknet53_270e_coco.yml](https://github.com/PaddlePaddle/PaddleDetection/blob/master/dygraph/configs/yolov3/yolov3_darknet53_270e_coco.yml) to train YOLOv3 on the COCO dataset, compared with this file, we modify some parameters as followed to conduct the training for vehicle detection:

* num_classes: 6
* anchors: [[8, 9], [10, 23], [19, 15], [23, 33], [40, 25], [54, 50], [101, 80], [139, 145], [253, 224]]
* nms/nms_top_k: 400
* nms/score_threshold: 0.005
* dataset_dir: dataset/vehicle

### 3. Accuracy

The accuracy of the model trained and evaluated on our private data is shown as followed:

AP at IoU=.50:.05:.95 is 0.545.

AP at IoU=.50 is 0.764.

### 4. Inference

Users can employ the model to conduct the inference:

```
export CUDA_VISIBLE_DEVICES=0
python -u tools/infer.py -c configs/vehicle/vehicle_yolov3_darknet.yml \
                         -o weights=https://paddledet.bj.bcebos.com/models/vehicle_yolov3_darknet.pdparams \
                         --infer_dir configs/vehicle/demo \
                         --draw_threshold 0.2 \
                         --output_dir configs/vehicle/demo/output
```

Some inference results are visualized below:

![](https://github.com/PaddlePaddle/PaddleDetection/tree/master/docs/images/VehicleDetection_001.jpeg)

![](https://github.com/PaddlePaddle/PaddleDetection/tree/master/docs/images/VehicleDetection_005.png)
