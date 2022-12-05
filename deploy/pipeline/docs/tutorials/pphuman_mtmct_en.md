English | [简体中文](pphuman_mtmct.md)

# Multi-Target Multi-Camera Tracking Module of PP-Human

Multi-target multi-camera tracking, or MTMCT, matches the identity of a person in different cameras based on the single-camera tracking. MTMCT is usually applied to the security system and the smart retailing.
The MTMCT module of PP-Human aims to provide a multi-target multi-camera pipleline which is simple, and efficient.

## How to Use

1. Download [REID model](https://bj.bcebos.com/v1/paddledet/models/pipeline/reid_model.zip) and unzip it to ```./output_inference```. For the MOT model, please refer to [mot description](./pphuman_mot.md).

2. In the MTMCT mode, input videos are required to be put in the same directory. set the REID "enable: True" in the infer_cfg_pphuman.yml. The command line is:
```python
python3 deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_pphuman.yml --video_dir=[your_video_file_directory] --device=gpu
```

3. Configuration can be modified in `./deploy/pipeline/config/infer_cfg_pphuman.yml`.

```python
python3 deploy/pipeline/pipeline.py
        --config deploy/pipeline/config/infer_cfg_pphuman.yml -o REID.model_dir=reid_best/
        --video_dir=[your_video_file_directory]
        --device=gpu
```

## Intorduction to the Solution

MTMCT module consists of the multi-target multi-camera tracking pipeline and the REID model.

1. Multi-Target Multi-Camera Tracking Pipeline

```

single-camera tracking[id+bbox]
        │
capture the target in the original image according to bbox——│
        │            │
    REID model      quality assessment (covered or not, complete or not, brightness, etc.)
        │            │
    [feature]        [quality]
        │            │
   datacollector—————│
        │
      sort out and filter features
        │
 calculate the similarity of IDs in the videos
        │
  make the IDs cluster together and rearrange them
```

2. The model solution is [reid-strong-baseline](https://github.com/michuanhaohao/reid-strong-baseline), with ResNet50 as the backbone.

Under the above circumstances, the REID model used in MTMCT integrates open-source datasets and compresses model features to 128-dimensional features to optimize the generalization. In this way, the actual generalization result becomes much better.

### Other Suggestions

- The provided REID model is obtained from open-source dataset training. It is recommended to add your own data to get a more powerful REID model, notably improving the MTMCT effect.
- The quality assessment is based on simple logic +OpenCV, whose effect is limited. If possible, it is advisable to conduct specific training on the quality assessment model.


### Example

- camera 1:
<div width="600" align="center">
  <img src="https://user-images.githubusercontent.com/22989727/205595795-fd859feb-8218-450f-a109-91c27713a662.gif"/>
</div>

- camera 2:
<div width="600" align="center">
  <img src="https://user-images.githubusercontent.com/22989727/205595826-18ab5f0e-a572-4950-a502-96e6eb904a1e.gif"/>
</div>


## Reference
```
@InProceedings{Luo_2019_CVPR_Workshops,
author = {Luo, Hao and Gu, Youzhi and Liao, Xingyu and Lai, Shenqi and Jiang, Wei},
title = {Bag of Tricks and a Strong Baseline for Deep Person Re-Identification},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {June},
year = {2019}
}

@ARTICLE{Luo_2019_Strong_TMM,
author={H. {Luo} and W. {Jiang} and Y. {Gu} and F. {Liu} and X. {Liao} and S. {Lai} and J. {Gu}},
journal={IEEE Transactions on Multimedia},
title={A Strong Baseline and Batch Normalization Neck for Deep Person Re-identification},
year={2019},
pages={1-1},
doi={10.1109/TMM.2019.2958756},
ISSN={1941-0077},
}
```
