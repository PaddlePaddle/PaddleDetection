[简体中文](./keypoint_detection.md) | English

# Customized Keypoint Detection

When applying keypoint detection algorithms in real practice, inevitably, we may need customization as we may dissatisfy with the current pre-trained model results, or the current keypoint detection cannot meet the actual demand, or we may want to add or replace the definition of keypoints and train a new keypoint detection model. This document will introduce how to customize the keypoint detection algorithm in PaddleDetection.

## Data Preparation

### Basic Process Description

PaddleDetection currently supports `COCO` and `MPII` annotation data formats. For detailed descriptions of these two data formats, please refer to the document [Keypoint Data Preparation](./../tutorials/data/PrepareKeypointDataSet.md). In this step, by using annotation tools such as Labeme, the corresponding coordinates are annotated according to the feature point serial numbers and then converted into the corresponding trainable annotation format. And we recommend `COCO` format.

### Merging datasets

To extend the training data, we can merge several different datasets together. But different datasets often have different definitions of key points. Therefore, the first step in merging datasets is to unify the point definitions of different datasets, and determine the benchmark points, i.e., the types of feature points finally learned by the model, and then adjust them according to the relationship between the point definitions of each dataset and the benchmark point definitions.

- Points in the benchmark point location: adjust the point number to make it consistent with the benchmark point location
- Points that are not in the benchmark points: discard
- Points in the dataset that are missing from the benchmark: annotate the marked points as "unannotated".

In [Key point data preparation](... /... /tutorials/data/PrepareKeypointDataSet.md), we provide a case illustration of how to merge the `COCO` dataset and the `AI Challenger` dataset and unify them as a benchmark point definition with `COCO` for your reference.

## Model Optimization

### Detection and tracking model optimization

In PaddleDetection, the keypoint detection supports Top-Down and Bottom-Up solutions. Top-Down first detects the main body and then detects the local key points. It has higher accuracy but will take a longer time as the number of detected objects increases.The Bottom-Up plan first detects the keypoints and then combines them with the corresponding parts. It is fast and its speed is independent of the number of detected objects. Its disadvantage is that the accuracy is relatively low. For details of the two solutions and the corresponding models, please refer to [Keypoint Detection Series Models](../../../configs/keypoint/README.md)

When using the Top-Down solution, the model's effects depend on the previous detection or tracking effect. If the pedestrian position cannot be accurately detected in the actual practice, the performance of the keypoint detection will be limited. If you encounter the above problem in actual application, please refer to [Customized Object Detection](./detection_en.md) and [Customized Multi-target tracking](./pphuman_mot_en.md) for optimization of the detection and tracking model.

### Iterate with scenario-compatible data

The currently released keypoint detection algorithm models are mainly iterated on open source datasets such as `COCO`/ `AI Challenger`, which may lack surveillance scenarios (angles, lighting and other factors), sports scenarios (more unconventional poses) that are more similar to the actual task. Training with data that more closely matches the actual task scenario can help improve the model's results.

### Iteration via pre-trained models

The data annotation of the keypoint model is complex, and using the model directly to train on the business dataset from scratch is often difficult to meet the demand. When used in practical projects, it is recommended to load the pre-trained weights, which usually improve the model accuracy significantly. Let's take `HRNet` as an example  with the following method:

```
python tools/train.py \
        -c configs/keypoint/hrnet/hrnet_w32_256x192.yml \
        -o pretrain_weights=https://paddledet.bj.bcebos.com/models/keypoint/hrnet_w32_256x192.pdparams
```

After loading the pre-trained model, the initial learning rate and the rounds of iterations can be reduced appropriately. It is recommended that the initial learning rate be 1/2 to 1/5 of the default configuration, and you can enable`--eval` to observe the change of AP values during the iterations.

## Data augmentation with occlusion

There are a lot of data in occlusion in keypoint tasks, including self-covered objects and occlusion between different objects.

1. Detection model optimization (only for Top-Down solutions)

Refer to [Target Detection Task Secondary Development](. /detection.md) to improve the detection model in complex scenarios.

2. Keypoint data augmentation

Augmentation of covered data in keypoint model training to improve model performance in such scenarios, please refer to [PP-TinyPose](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/configs/keypoint/tiny_pose/)

### Smooth video prediction

The keypoint model is trained and predicted on the basis of image, and video input is also predicted by splitting the video into frames. Although the content is mostly similar between frames, small differences may still lead to large changes in the output of the model. As a result of that, although the predicted coordinates are roughly correct, there may be jitters in the visual effect.

By adding a smoothing filter process, the performance of the video output can be effectively improved by combining the predicted results of each frame and the historical results. For this part, please see [Filter Smoothing](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.6/deploy/python/det_keypoint_unite_infer.py#L206).

## Add or modify keypoint definition

### Data Preparation

Complete the data preparation according to the previous instructions and place it under `{root of PaddleDetection}/dataset`.

<details>
<summary><b> Examples of annotation file</b></summary>

```
self_dataset/
├── train_coco_joint.json # training set annotation file
├── val_coco_joint.json # Validation set annotation file
├── images/ # Store the image files
    ├── 0.jpg
    ├── 1.jpg
    ├── 2.jpg  
```

Notable changes as follows:

```
{
    "images": [
        {
            "file_name": "images/0.jpg",
            "id": 0, # image id, id cannotdo not repeat
            "height": 1080,
            "width": 1920
        },
        {
            "file_name": "images/1.jpg",
            "id": 1,
            "height": 1080,
            "width": 1920
        },
        {
            "file_name": "images/2.jpg",
            "id": 2,
            "height": 1080,
            "width": 1920
        },
    ...

    "categories": [
        {
            "supercategory": "person",
            "id": 1,
            "name": "person",
            "keypoints": [ # the name of the point serial number
                "point1",
                "point2",
                "point3",
                "point4",
                "point5",
            ],
            "skeleton": [ # Skeleton composed of points, not necessary for training
                [
                    1,
                    2
                ],
                [
                    1,
                    3
                ],
                [
                    2,
                    4
                ],
                [
                    3,
                    5
                ]
            ]
    ...

    "annotations": [
        {
            {
            "category_id": 1, # The category to which the instance belongs
            "num_keypoints": 3, # the number of marked points of the instance
              "bbox": [         # location of detection box,format is x, y, w, h
                799,
                575,
                55,
                185
            ],
            # N*3 list of x, y, v.
            "keypoints": [  
                807.5899658203125,
                597.5455322265625,
                2,
                0,  
                0,
                0, # unlabeled points noted as 0, 0, 0
                805.8563232421875,
                592.3446655273438,
                2,
                816.258056640625,
                594.0783081054688,
                2,
                0,
                0,
                0
            ]
            "id": 1, # the id of the instance, id cannot repeat
            "image_id": 8, # The id of the image where the instance is located, repeatable. This represents the presence of multiple objects on a single image
"iscrowd": 0, # covered or not, when the value is 0, it will participate in training
            "area": 10175 # the area occupied by the instance, can be simply taken as w * h. Note that when the value is 0, it will be skipped, and if it is too small, it will be ignored in eval

    ...
```

### Settings of configuration file

In the configuration file, refer to [config yaml configuration](... /... /tutorials/KeyPointConfigGuide_cn.md) for more details . Take [HRNet model configuration](... /... /... /configs/keypoint/hrnet/hrnet_w32_256x192.yml) as an example, we need to focus on following contents:

<details>
<summary><b> Example of configuration</b></summary>

```
use_gpu: true
log_iter: 5
save_dir: output
snapshot_epoch: 10
weights: output/hrnet_w32_256x192/model_final
epoch: 210
num_joints: &num_joints 5 # The number of predicted points matches the number of defined points
pixel_std: &pixel_std 200
Metric. keyPointTopDownCOCOEval
num_classes: 1  
train_height: &train_height 256
train_width: &train_width 192
trainsize: &trainsize [*train_width, *train_height].
hmsize: &hmsize [48, 64].
flip_perm: &flip_perm [[1, 2], [3, 4]]. # Note that only points that are mirror-symmetric are recorded here.

...

# Ensure that dataset_dir + anno_path can correctly locate the annotation file
# Ensure that dataset_dir + image_dir + image path in annotation file can correctly locate the image.
TrainDataset:
  !KeypointTopDownCocoDataset
    image_dir: images
    anno_path: train_coco_joint.json
    dataset_dir: dataset/self_dataset
    num_joints: *num_joints
    trainsize. *trainsize
    pixel_std: *pixel_std
    use_gt_box: true


Evaluate the dataset.
  !KeypointTopDownCocoDataset
    image_dir: images
    anno_path: val_coco_joint.json
    dataset_dir: dataset/self_dataset
    bbox_file: bbox.json
    num_joints: *num_joints
    trainsize. *trainsize
    pixel_std: *pixel_std
    use_gt_box: true
    image_thre: 0.0
```

### Model Training and Evaluation

#### Model Training

Run the following command to start training:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m paddle.distributed.launch tools/train.py -c configs/keypoint/hrnet/hrnet_w32_256x192.yml
```

#### Model Evaluation

After training the model, you can evaluate the model metrics by running the following commands:

```
python3 tools/eval.py -c configs/keypoint/hrnet/hrnet_w32_256x192.yml
```

### Model Export and Inference

#### Top-Down model deployment

```
#Export keypoint model
python tools/export_model.py -c configs/keypoint/hrnet/hrnet_w32_256x192.yml -o weights={path_to_your_weights}

#detector detection + keypoint top-down model co-deployment（for top-down solutions only）
python deploy/python/det_keypoint_unite_infer.py --det_model_dir=output_inference/ppyolo_r50vd_dcn_2x_coco/ --keypoint_model_dir=output_inference/hrnet_w32_256x192/ --video_file=../video/xxx.mp4  --device=gpu
```
