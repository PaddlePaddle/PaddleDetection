English | [简体中文](README_cn.md)
# PaddleDetection applied for specific scenarios

We provide some models implemented by PaddlePaddle to detect objects in specific scenarios, users can download the models and use them in these scenarios.

| Task                 | Algorithm | Box AP | Download                                                                                | Configs |
|:---------------------|:---------:|:------:| :-------------------------------------------------------------------------------------: |:------:|
| Pedestrian Detection |  YOLOv3  |  51.8  | [model](https://paddledet.bj.bcebos.com/models/pedestrian_yolov3_darknet.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/pedestrian/pedestrian_yolov3_darknet.yml) |

## Pedestrian Detection

The main applications of pedetestrian detection include intelligent monitoring. In this scenary, photos of pedetestrians are taken by surveillance cameras in public areas, then pedestrian detection are conducted on these photos.

### 1. Network

The network for detecting vehicles is YOLOv3, the backbone of which is Dacknet53.

### 2. Configuration for training

PaddleDetection provides users with a configuration file [yolov3_darknet53_270e_coco.yml](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/configs/yolov3/yolov3_darknet53_270e_coco.yml) to train YOLOv3 on the COCO dataset, compared with this file, we modify some parameters as followed to conduct the training for pedestrian detection:

* num_classes: 1
* dataset_dir: dataset/pedestrian

### 3. Accuracy

The accuracy of the model trained and evaluted on our private data is shown as followed:

AP at IoU=.50:.05:.95 is 0.518.

AP at IoU=.50 is 0.792.

### 4. Inference

Users can employ the model to conduct the inference:

```
export CUDA_VISIBLE_DEVICES=0
python -u tools/infer.py -c configs/pedestrian/pedestrian_yolov3_darknet.yml \
                         -o weights=https://paddledet.bj.bcebos.com/models/pedestrian_yolov3_darknet.pdparams \
                         --infer_dir configs/pedestrian/demo \
                         --draw_threshold 0.3 \
                         --output_dir configs/pedestrian/demo/output
```

Some inference results are visualized below:

![](../../docs/images/PedestrianDetection_001.png)

![](../../docs/images/PedestrianDetection_004.png)
