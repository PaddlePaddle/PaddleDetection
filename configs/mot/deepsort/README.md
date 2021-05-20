English | [简体中文](README_cn.md)

# DeepSORT (Simple Online and Realtime Tracking with a Deep Association Metric)

## Table of Contents
- [Introduction](#Introduction)
- [Model Zoo](#Model_Zoo)
- [Getting Start](#Getting_Start)
- [Citations](#Citations)

## Introduction
[DeepSORT](https://arxiv.org/abs/1812.00442) (Deep Cosine Metric Learning SORT) extends the original [SORT](https://arxiv.org/abs/1703.07402) (Simple Online and Realtime Tracking) algorithm to integrate appearance information based on a deep appearance descriptor. It adds a CNN model to extract features in image of human part bounded by a detector. Here we use `JDE` as detection model to generate boxes, and select `PCBPyramid` as the ReID model. We also support loading the boxes from saved detection result files.

## Model Zoo

### DeepSORT on MOT-16 training set

| backbone  | input shape | MOTA | IDF1 |  IDS  |   FP  |   FN  |   FPS  | Detector | ReID | config |
| :---------| :------- | :----: | :----: | :--: | :----: | :---: | :---: |:-------: | :---: | :---: |
| DarkNet53 | 1088x608 |  72.2  |  60.5  | 998  |  8054  | 21644 |  5.07 |[JDE](https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_1088x608.pdparams)| [ReID](https://paddledet.bj.bcebos.com/models/mot/deepsort_pcb_pyramid_r101.pdparams)|[config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/mot/deepsort/deepsort_pcb_pyramid_r101.yml) |

**Notes:**
 DeepSORT does not need to train on MOT dataset, only used for evaluation. Before DeepSORT evaluation, you should get detection results by a detection model first, here we use JDE, and then prepare them like this:
```
det_results_dir
   |——————MOT16-02.txt
   |——————MOT16-04.txt
   |——————MOT16-05.txt
   |——————MOT16-09.txt
   |——————MOT16-10.txt
   |——————MOT16-11.txt
   |——————MOT16-13.txt
```
Each txt is the detection result of all the pictures extracted from each video, and each line describes a bounding box with the following format:
```
[frame_id],[identity],[bb_left],[bb_top],[width],[height],[conf],[x],[y],[z]
```
**Notes:**
- `frame_id` is the frame number of the image
- `identity` is the object id using default value `-1`
- `bb_left` is the X coordinate of the left bound of the object box
- `bb_top` is the Y coordinate of the upper bound of the object box
- `width,height` is the pixel width and height
- `conf` is the object score with default value `1` (the results had been filtered out according to the detection score threshold)
- `x,y,z` are used in 3D, default to `-1` in 2D.

## Getting Start

### 1. Evaluate a detector to get detection results

```bash
# use weights released in PaddleDetection model zoo
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/jde/jde_darknet53_30e_1088x608_track.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_1088x608.pdparams

# use saved checkpoint after training
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/jde/jde_darknet53_30e_1088x608_track.yml -o weights=output/jde_darknet53_30e_1088x608/model_final.pdparams
```

### 2. Tracking

```bash
# track the objects by loading detected result files
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/deepsort/deepsort_pcb_pyramid_r101.yml --det_results_dir {your detection results}
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
