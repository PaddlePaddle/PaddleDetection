[简体中文](./idbased_clas.md) | English

# Development for Action Recognition Based on Classification with Human ID

## Environmental Preparation
The model of action recognition based on classification with human id is trained with [PaddleClas](https://github.com/PaddlePaddle/PaddleClas). Please refer to [Install PaddleClas](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/en/installation/install_paddleclas_en.md) to complete the environment installation for subsequent model training and usage processes.

## Data Preparation

The model of action recognition based on classification with human id directly recognizes the image frames of video, so the model training process is same with the usual image classification model.

### Dataset Download

The action recognition of making phone calls is trained on the public dataset [UAV-Human](https://github.com/SUTDCV/UAV-Human). Please fill in the relevant application materials through this link to obtain the download link.

The RGB video in this dataset is included in the `UAVHuman/ActionRecognition/RGBVideos` path, and the file name of each video is its annotation information.

### Image Processing for Training and Validation
According to the video file name, in which the `A` field (i.e. action) related to action recognition, we can find the action type of the video data that we expect to recognize.
- Positive sample video: Taking phone calls as an example, we just need to find the file containing `A024`.
- Negative sample video: All videos except the target action.

In view of the fact that there will be much redundancy when converting video data into images, for positive sample videos, we sample at intervals of 8 frames, and use the pedestrian detection model to process it into a half-body image (take the upper half of the detection frame, that is, `img = img[: H/2, :, :]`). The image sampled from the positive sample video is regarded as a positive sample, and the sampled image from the negative sample video is regarded as a negative sample.

**Note**: The positive sample video does not completely are the action of making a phone call. There will be some redundant actions at the beginning and end of the video, which need to be removed.


### Preparation for Annotation File
The model of action recognition based on classification with human id is trained with [PaddleClas](https://github.com/PaddlePaddle/PaddleClas). Thus the model trained with this scheme needs to prepare the desired image data and corresponding annotation files. Please refer to [Image Classification Datasets](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/en/data_preparation/classification_dataset_en.md) to prepare the data. An example of an annotation file is as follows, where `0` and `1` are the corresponding categories of the image:

```
    # Each line uses "space" to separate the image path and label
    train/000001.jpg 0
    train/000002.jpg 0
    train/000003.jpg 1
    ...
```

Additionally, the label file `phone_label_list.txt` helps map category numbers to specific type names:
```
0 make_a_phone_call # type 0
1 normal # type 1
```

After the above content finished, place it to the `dataset` directory, the file structure is as follow:
```
data/
├── images  # All images
├── phone_label_list.txt # Label file
├── phone_train_list.txt # Training list, including pictures and their corresponding types
└── phone_val_list.txt   # Validation list, including pictures and their corresponding types
```

## Model Optimization

### Detection-Tracking Model Optimization
The performance of action recognition based on classification with human id depends on the pre-order detection and tracking models. If the pedestrian location cannot be accurately detected in the actual scene, or it is difficult to correctly assign the person ID between different frames, the performance of the action recognition part will be limited. If you encounter the above problems in actual use, please refer to [Secondary Development of Detection Task](../detection_en.md) and [Secondary Development of Multi-target Tracking Task](../pphuman_mot_en.md) for detection/track model optimization.


### Half-Body Prediction
In the action of making a phone call, the action classification can be achieved through the upper body image. Therefore, during the training and prediction process, the image is changed from the pedestrian full-body to half-body.

## Add New Action

### Data Preparation
Referring to the previous introduction, complete the data preparation part and place it under `{root of PaddleClas}/dataset`:

```
data/
├── images  # All images
├── label_list.txt # Label file
├── train_list.txt # Training list, including pictures and their corresponding types
└── val_list.txt   # Validation list, including pictures and their corresponding types
```
Where the training list and validation list file are as follow:
```
    # Each line uses "space" to separate the image path and label
    train/000001.jpg 0
    train/000002.jpg 0
    train/000003.jpg 1
    train/000004.jpg 2   # For the newly added categories, simply fill in the corresponding category number.

`label_list.txt` should give name of the extension type:
```
0 make_a_phone_call  # class 0
1 Your New Action    # class 1
 ...
n normal             # class n
```
    ...
```

### Configuration File Settings
The [training configuration file] (https://github.com/PaddlePaddle/PaddleClas/blob/develop/ppcls/configs/practical_models/PPHGNet_tiny_calling_halfbody.yaml) has been integrated in PaddleClas. The settings that need to be paid attention to are as follows:

```yaml
# model architecture
Arch:
  name: PPHGNet_tiny
  class_num: 2       # Corresponding to the number of action categories

  ...

# Please correctly set image_root and cls_label_path to ensure that the image_root + image path in cls_label_path can access the image correctly
DataLoader:
  Train:
    dataset:
      name: ImageNetDataset
      image_root: ./dataset/
      cls_label_path: ./dataset/phone_train_list_halfbody.txt

      ...

Infer:
  infer_imgs: docs/images/inference_deployment/whl_demo.jpg
  batch_size: 1
  transforms:
    - DecodeImage:
        to_rgb: True
        channel_first: False
    - ResizeImage:
        size: 224
    - NormalizeImage:
        scale: 1.0/255.0
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
        order: ''
    - ToCHWImage:
  PostProcess:
    name: Topk
    topk: 2                                           # Display the number of topks, do not exceed the total number of categories
    class_id_map_file: dataset/phone_label_list.txt   # path of label_list.txt
```

### Model Training And Evaluation
#### Model Training
Start training with the following command:
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/practical_models/PPHGNet_tiny_calling_halfbody.yaml \
        -o Arch.pretrained=True
```
where `Arch.pretrained=True` is to use pretrained weights to help with training.

#### Model Evaluation
After training the model, use the following command to evaluate the model metrics.
```bash
python3 tools/eval.py \
    -c ./ppcls/configs/practical_models/PPHGNet_tiny_calling_halfbody.yaml \
    -o Global.pretrained_model=output/PPHGNet_tiny/best_model
```
Where `-o Global.pretrained_model="output/PPHGNet_tiny/best_model"` specifies the path where the current best weight is located. If other weights are needed, just replace the corresponding path.

#### Model Export
For the detailed introduction of model export, please refer to [here](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/en/inference_deployment/export_model_en.md#2-export-classification-model)
You can refer to the following steps:

```python
python tools/export_model.py
    -c ./PPHGNet_tiny_calling_halfbody.yaml \
    -o Global.pretrained_model=./output/PPHGNet_tiny/best_model \
    -o Global.save_inference_dir=./output_inference/PPHGNet_tiny_calling_halfbody
```

Then rename the exported model and add the configuration file to suit the usage of PP-Human.
```bash
cd ./output_inference/PPHGNet_tiny_calling_halfbody

mv inference.pdiparams model.pdiparams
mv inference.pdiparams.info model.pdiparams.info
mv inference.pdmodel model.pdmodel

# Download configuration file for inference
wget https://bj.bcebos.com/v1/paddledet/models/pipeline/infer_configs/PPHGNet_tiny_calling_halfbody/infer_cfg.yml
```

At this point, this model can be used in PP-Human.

### Custom Action Output
In the model of action recognition based on classification with human id, the task is defined as a picture-level classification task of corresponding person. The type of the corresponding classification is finally regarded as the action type of the current stage. Therefore, on the basis of completing the training and deployment of the custom model, it is also necessary to convert the classification model results to the final action recognition results as output, and the displayed result of the visualization should be modified.

Please modify the [postprocessing function](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/deploy/pipeline/pphuman/action_infer.py#L509).

The core code are:
```python
# Get the highest score output of the classification model
cls_id_res = 1
cls_score_res = -1.0
for cls_id in range(len(cls_result[idx])):
    score = cls_result[idx][cls_id]
    if score > cls_score_res:
        cls_id_res = cls_id
        cls_score_res = score

# Current now,  class 0 is positive, class 1 is negative.
if cls_id_res == 1 or (cls_id_res == 0 and
                       cls_score_res < self.threshold):
    # If the classification result is not the target action or its confidence does not reach the threshold,
    # determine the action type of the current frame according to the historical results
    history_cls, life_remain, history_score = self.result_history.get(
        tracker_id, [1, self.frame_life, -1.0])
    cls_id_res = history_cls
    cls_score_res = 1 - cls_score_res
    life_remain -= 1
    if life_remain <= 0 and tracker_id in self.result_history:
        del (self.result_history[tracker_id])
    elif tracker_id in self.result_history:
        self.result_history[tracker_id][1] = life_remain
    else:
        self.result_history[
            tracker_id] = [cls_id_res, life_remain, cls_score_res]
else:
    # If the classification result belongs to the target action, use the result and record it in the historical result
    self.result_history[
        tracker_id] = [cls_id_res, self.frame_life, cls_score_res]

    ...
```

#### Modify Visual Output
At present, ID-based action recognition is displayed based on the results of action recognition and predefined category names. For the detail, please refer to [here](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/deploy/pipeline/pipeline.py#L1024-L1043). If the custom action needs to be modified to another display name, please modify it accordingly to output the corresponding result.
