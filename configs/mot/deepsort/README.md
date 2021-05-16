English | [简体中文](README_cn.md)

# DeepSORT (Simple Online and Realtime Tracking with a Deep Association Metric)

## Table of Contents
- [Introduction](#Introduction)
- [Model Zoo](#Model_Zoo)
- [Getting Start](#Getting_Start)

## Introduction
[DeepSORT](https://arxiv.org/abs/1812.00442) is basicly the same with SORT but added a CNN model to extract features in image of human part bounded by a detector. We use JDE as detection model to generate boxes, and select `PCBPyramid` as the ReID model. We also support loading the boxes from saved detection result files.

## Model Zoo

### DeepSORT on MOT-16 training set

| backbone  | input shape  | MOTA   | IDF1   |  IDS  |   FP  |   FN  |   FPS  | Detector | ReID | config |
| :---------| :------- | :----: | :----: | :--: | :----: | :---: | :---: |:---: | :---: | :---: |
| DarkNet53 | 1088x608 |  72.2  |  60.3  | 998  |  8055  | 21631 |  3.28 |[JDE](https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_1088x608.pdparams)| [ReID](https://paddledet.bj.bcebos.com/models/mot/deepsort_pcb_pyramid_r101.pdparams)|[config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/mot/deepsort/deepsort_pcb_pyramid_r101.yml) |

**Notes:**
 DeepSORT does not need to train, only used for evaluation. Before DeepSORT evaluation, you should get detection results by a detection model first, here we use JDE, and then prepare them like this:
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

## Getting Start

### 1. Evaluate a detector to get detection results

```bash
# use weights released in PaddleDetection model zoo
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/jde/jde_darknet53_30e_1088x608_track.yml -o metric=MOT weights=https://paddledet.bj.bcebos.com/models/mot/jde_darknet53_30e_1088x608.pdparams

# use saved checkpoint after training
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/jde/jde_darknet53_30e_1088x608_track.yml -o metric=MOT weights=output/jde_darknet53_30e_1088x608/model_final
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
