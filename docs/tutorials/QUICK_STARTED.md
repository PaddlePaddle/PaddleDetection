English | [简体中文](QUICK_STARTED_cn.md)

# Quick Start
In order to enable users to experience PaddleDetection and produce models in a short time, this tutorial introduces the pipeline to get a decent object detection model by finetuning on a small dataset in 10 minutes only. In practical applications, it is recommended that users select a suitable model configuration file for their specific demand.

- **Set GPU**


```bash
export CUDA_VISIBLE_DEVICES=0
```

## Inference Demo with Pre-trained Models

```
# predict an image using PP-YOLO
python tools/infer.py -c configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml -o use_gpu=true weights=https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_1x_coco.pdparams --infer_img=demo/000000014439.jpg
```

the result：

![](../images/000000014439.jpg)


## Data preparation
The Dataset is [Kaggle dataset](https://www.kaggle.com/andrewmvd/road-sign-detection) ，including 877 images and 4 data categories: crosswalk, speedlimit, stop, trafficlight. The dataset is divided into training set (701 images) and test set (176 images)，[download link](https://paddlemodels.bj.bcebos.com/object_detection/roadsign_voc.tar).

```
# Note: this command could skip and
# the dataset will be dowloaded automatically at the stage of training.
python dataset/roadsign_voc/download_roadsign_voc.py
```

## Training & Evaluation & Inference
### 1、Training
```
# It will takes about 10 minutes on 1080Ti and 1 hour on CPU
# -c set configuration file
# -o overwrite the settings in the configuration file
# --eval Evaluate while training, and a model named best_model.pdmodel with the most evaluation results will be automatically saved


python tools/train.py -c configs/yolov3/yolov3_mobilenet_v1_roadsign.yml --eval -o use_gpu=true
```

If you want to observe the loss change curve in real time through VisualDL, add --use_vdl=true to the training command, and set the log save path through --vdl_log_dir.

**Note: VisualDL need Python>=3.5**

Please install [VisualDL](https://github.com/PaddlePaddle/VisualDL) first

```
python -m pip install visualdl -i https://mirror.baidu.com/pypi/simple
```

```
python -u tools/train.py -c configs/yolov3/yolov3_mobilenet_v1_roadsign.yml \
                        --use_vdl=true \
                        --vdl_log_dir=vdl_dir/scalar \
                        --eval
```
View the change curve in real time through the visualdl command:
```
visualdl --logdir vdl_dir/scalar/ --host <host_IP> --port <port_num>
```

### 2、Evaluation
```
# Evaluate best_model by default
# -c set config file
# -o overwrite the settings in the configuration file

python tools/eval.py -c configs/yolov3/yolov3_mobilenet_v1_roadsign.yml -o use_gpu=true
```

The final mAP should be around 0.85. The dataset is small so the precision may vary a little after each training.


### 3、Inference
```
# -c set config file
# -o overwrite the settings in the configuration file
# --infer_img image path
# After the prediction is over, an image of the same name with the prediction result will be generated in the output folder

python tools/infer.py -c configs/yolov3/yolov3_mobilenet_v1_roadsign.yml -o use_gpu=true --infer_img=demo/road554.png
```

The result is as shown below：

![](../images/road554.png)
