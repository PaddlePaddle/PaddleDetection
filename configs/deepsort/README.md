# Deep SORT(Simple Online and Realtime Tracking with a Deep Association Metric)

## Table of Contents
- [Introduction](#Introduction)
- [Model Zoo](#Model_Zoo)
- [Getting Start](#Getting_Start)
- [Appendix](#Appendix)

## Introduction
[Deep SORT](https://arxiv.org/abs/2007.12099) is basicly the same with sort but added a CNN model to extract features in image of human part bounded by a detector. We use YOLOv3 which is trained on pedestrian dataset to generate boxes instead of FasterRCNN, and select 'PCB_plus_dropout_pyramid' as the ReID model. Meanwhile, we support to load the boxes from result files instead of the detector.

## Model Zoo

### Deep SORT on MOT-16 training set

| 骨架网络   | 输入尺寸  | MOTA   | IDF1   |  IDS  |   FP  |   FN  |   FPS  | detector |  ReID    | 配置文件 |
| :---------| :------- | :----: | :----: | :--: | :----: | :---: | :---: |:---: | :---: | :---: |
| DarkNet53 | 1088x608 |  50.1  |  52.3  | 393  |  3229  | 51334 |  3.13 |[下载链接]()| [下载链接]()|[配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/deepsort/deepsort_pcb_tracker_1088x608.yml) |

## Getting Start

### 1. Training of detector

Training YOLOv3 on 8 GPUs with following command(all commands should be run under PaddleDetection dygraph directory as default)

```bash
python -m paddle.distributed.launch --log_dir=./yolov3_dygraph/ --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/pedestrian/pedestrian_yolov3_darknet.yml
```

### 2. Evaluation of detector

Evaluating YOLOv3 on COCO val2017 dataset in single GPU with following commands:
```bash
# use weights released in PaddleDetection model zoo
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/pedestrian/pedestrian_yolov3_darknet.yml -o weights=https://paddledet.bj.bcebos.com/models/pedestrian_yolov3_darknet.pdparams

# use saved checkpoint in training
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/pedestrian/pedestrian_yolov3_darknet.yml -o weights=output/pedestrian_yolov3_darknet/model_final
```

### 3. Tracking
Tracking the multiple objdect in the images or video.
```bash
# detect and track the objects
CUDA_VISIBLE_DEVICES=0 python3.7 tools/eval_mot.py -c configs/deepsort/deepsort_pcb_tracker_1088x608.yml -o use_gpu=true --model_type deepsort

# track the objects by loading detected result file
CUDA_VISIBLE_DEVICES=0 python3.7 tools/eval_mot.py -c configs/deepsort/deepsort_pcb_tracker_1088x608.yml -o use_gpu=true --model_type deepsort --det_dir ./result_txts

# detect and track the objects, then save the result as images
CUDA_VISIBLE_DEVICES=0 python3.7 tools/eval_mot.py -c configs/deepsort/deepsort_pcb_tracker_1088x608.yml -o use_gpu=true --model_type deepsort --save_images

# detect and track the objects, then save the result as a video
CUDA_VISIBLE_DEVICES=0 python3.7 tools/eval_mot.py -c configs/deepsort/deepsort_pcb_tracker_1088x608.yml -o use_gpu=true --model_type deepsort --save_videos

# detect and track the objects by loading video
CUDA_VISIBLE_DEVICES=0 python3.7 tools/infer_mot.py -c configs/deepsort/deepsort_yolov3_darknet53_pcb_tracker_1088x608.yml -o use_gpu=true --video_file ./MOT16-05.mp4 --model_type deepsort
```

## Citations
```
@inproceedings{Wojke2017simple,
  title={Simple Online and Realtime Tracking with a Deep Association Metric},
  author={Wojke, Nicolai and Bewley, Alex and Paulus, Dietrich},
  booktitle={2017 IEEE International Conference on Image Processing (ICIP)},
  year={2017},
  pages={3645--3649},
  organization={IEEE},
  doi={10.1109/ICIP.2017.8296962}
}

@inproceedings{Wojke2018deep,
  title={Deep Cosine Metric Learning for Person Re-identification},
  author={Wojke, Nicolai and Bewley, Alex},
  booktitle={2018 IEEE Winter Conference on Applications of Computer Vision (WACV)},
  year={2018},
  pages={748--756},
  organization={IEEE},
  doi={10.1109/WACV.2018.00087}
}
```
