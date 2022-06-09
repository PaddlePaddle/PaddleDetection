[简体中文](PrepareKeypointDataSet_cn.md) | English

# How to prepare dataset?
## Table of Contents
- [COCO](#COCO)
- [MPII](#MPII)
- [Training for other dataset](#Training_for_other_dataset)

## COCO
### Preperation for COCO dataset
We provide a one-click script to automatically complete the download and preparation of the COCO2017 dataset. Please refer to [COCO Download](https://github.com/PaddlePaddle/PaddleDetection/blob/f0a30f3ba6095ebfdc8fffb6d02766406afc438a/docs/tutorials/PrepareDataSet.md#COCO%E6%95%B0%E6%8D%AE).

### Description for COCO dataset（Keypoint):
In COCO, the indexes and corresponding keypoint name are:
```
COCO keypoint indexes:
        0: 'nose',
        1: 'left_eye',
        2: 'right_eye',
        3: 'left_ear',
        4: 'right_ear',
        5: 'left_shoulder',
        6: 'right_shoulder',
        7: 'left_elbow',
        8: 'right_elbow',
        9: 'left_wrist',
        10: 'right_wrist',
        11: 'left_hip',
        12: 'right_hip',
        13: 'left_knee',
        14: 'right_knee',
        15: 'left_ankle',
        16: 'right_ankle'
```
Being different from detection task, the annotation files for keyPoint task are `person_keypoints_train2017.json` and `person_keypoints_val2017.json`. In these two json files, the terms `info`、`licenses` and `images` are same with detection task. However, the `annotations` and `categories` are different.

In `categories`, in addition to the category, there are also the names of the keypoints and the connectivity among them.

In `annotations`, the ID and image of each instance are annotated, as well as segmentation information and keypoint information. Among them, terms related to the keypoints are:
- `keypoints`: `[x1,y1,v1 ...]`, which is a `List` with length 17*3=51. Each combination represents the coordinates and visibility of one keypoint. `v=0, x=0, y=0` indicates this keypoint is not visible and unlabeled. `v=1` indicates this keypoint is labeled but not visible. `v=2` indicates this keypoint is labeled and visible.
- `bbox`: `[x1,y1,w,h]`, the bounding box of this instance.
- `num_keypoints`: the number of labeled keypoints of this instance.


## MPII
### Preperation for MPII dataset
Please download MPII dataset images and corresponding annotation files from [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/#download), and save them to `dataset/mpii`.  You can use [mpii_annotations](https://download.openmmlab.com/mmpose/datasets/mpii_annotations.tar), which are already converted to `.json`.  The directory structure will be shown as:
```
mpii
|── annotations
|   |── mpii_gt_val.mat
|   |── mpii_test.json
|   |── mpii_train.json
|   |── mpii_trainval.json
|   `── mpii_val.json
`── images
    |── 000001163.jpg
    |── 000003072.jpg
```
### Description for MPII dataset
In MPII, the indexes and corresponding keypoint name are:
```
MPII keypoint indexes:
        0: 'right_ankle',
        1: 'right_knee',
        2: 'right_hip',
        3: 'left_hip',
        4: 'left_knee',
        5: 'left_ankle',
        6: 'pelvis',
        7: 'thorax',
        8: 'upper_neck',
        9: 'head_top',
        10: 'right_wrist',
        11: 'right_elbow',
        12: 'right_shoulder',
        13: 'left_shoulder',
        14: 'left_elbow',
        15: 'left_wrist',
```
The following example takes a parsed annotation information to illustrate the content of the annotation, each annotation information represents a person instance:
```
{
    'joints_vis': [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    'joints': [
        [-1.0, -1.0],
        [-1.0, -1.0],
        [-1.0, -1.0],
        [-1.0, -1.0],
        [-1.0, -1.0],
        [-1.0, -1.0],
        [-1.0, -1.0],
        [1232.0, 288.0],
        [1236.1271, 311.7755],
        [1181.8729, -0.77553],
        [692.0, 464.0],
        [902.0, 417.0],
        [1059.0, 247.0],
        [1405.0, 329.0],
        [1498.0, 613.0],
        [1303.0, 562.0]
    ],
    'image': '077096718.jpg',
    'scale': 9.516749,
    'center': [1257.0, 297.0]
}
```
- `joints_vis`: indicates whether the 16 keypoints are labeled respectively, if it is 0, the corresponding coordinate will be `[-1.0, -1.0]`.
- `joints`: the coordinates of 16 keypoints.
- `image`: image file which this instance belongs to.
- `center`: the coordinate of person instance center, which is used to locate instance in the image.
- `scale`: scale of the instance, corresponding to 200px.


## Training for other dataset
Here, we take `AI Challenger` dataset as example, to show how to align other datasets to `COCO` and add them into training of keypoint models.

In `AI Challenger`, the indexes and corresponding keypoint name are:
```
AI Challenger Description:
        0: 'Right Shoulder',
        1: 'Right Elbow',
        2: 'Right Wrist',
        3: 'Left Shoulder',
        4: 'Left Elbow',
        5: 'Left Wrist',
        6: 'Right Hip',
        7: 'Right Knee',
        8: 'Right Ankle',
        9: 'Left Hip',
        10: 'Left Knee',
        11: 'Left Ankle',
        12: 'Head top',
        13: 'Neck'
```
1. Align the indexes of the `AI Challenger` keypoint to be consistent with `COCO`. For example, the index of `Right Shoulder` should be adjusted from `0` to `13`.
2. Unify the flags whether the keypoint is labeled/visible. For example, `labeled and visible` in `AI Challenger` needs to be adjusted from `1` to `2`.
3. In this proprocess, we discard the unique keypoints in this dataset (like `Neck`). For keypoints not in this dataset but in `COCO` (like `left_eye`), we set `v=0, x=0, y=0` to indicate these keypoints are not labeled.
4. To avoid the problem of ID duplication in different datasets, the `image_id` and `annotation id` need to be rearranged.
5. Rewrite the image path `file_name`, to make sure images can be accessed correctly.

We also provide an [annotation file](https://bj.bcebos.com/v1/paddledet/data/keypoint/aic_coco_train_cocoformat.json) combining `COCO` trainset and `AI Challenger` trainset.
