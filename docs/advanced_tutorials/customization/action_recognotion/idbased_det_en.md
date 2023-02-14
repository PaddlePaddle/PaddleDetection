[简体中文](./idbased_det.md) | English

# Development for Action Recognition Based on Detection with Human ID

## Environmental Preparation
The model of action recognition based on detection with human id is trained with [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection). Please refer to [Installation](../../../tutorials/INSTALL.md) to complete the environment installation for subsequent model training and usage processes.

## Data Preparation

The model of action recognition based on detection with human id directly recognizes the image frames of video, so the model training process is same with preparation process of general detection model. For details, please refer to [Data Preparation for Detection](../../../tutorials/data/PrepareDetDataSet_en.md). Please process image and annotation of data into one of the formats PaddleDetection supports.

**Note**: In the actual prediction process, a single person image is used for prediction. So it is recommended to crop the image into a single person image during the training process, and label the cigarette detection bounding box to improve the accuracy.


## Model Optimization
### Detection-Tracking Model Optimization
The performance of action recognition based on detection with human id depends on the pre-order detection and tracking models. If the pedestrian location cannot be accurately detected in the actual scene, or it is difficult to correctly assign the person ID between different frames, the performance of the action recognition part will be limited. If you encounter the above problems in actual use, please refer to [Secondary Development of Detection Task](../detection_en.md) and [Secondary Development of Multi-target Tracking Task](../pphuman_mot_en.md) for detection/track model optimization.


### Larger resolution
The detection of cigarette is a typical small target detection problem from the monitoring perspective. Using a larger resolution can help improve the overall performance of the model.

### Pretrained model
The pretrained model under the small target scene dataset VisDrone is used for training, and the mAP of the model is increased from 38.1 to 39.7.

## Add New Action
### Data Preparation
please refer to [Data Preparation for Detection](../../../tutorials/data/PrepareDetDataSet_en.md) to complete the data preparation part.

When finish this step, the path will look like:
```
dataset/smoking
├── smoking # all images
│   ├── 1.jpg
│   ├── 2.jpg
├── smoking_test_cocoformat.json # Validation file
├── smoking_train_cocoformat.json # Training file
```

Taking the `COCO` format as an example, the content of the completed json annotation file is as follows:

```json
# The "images" field contains the path, id and corresponding width and height information of the images.
  "images": [
    {
      "file_name": "smoking/1.jpg",
      "id": 0,    # Here id is the picture id serial number, do not duplicate
      "height": 437,
      "width": 212
    },
    {
      "file_name": "smoking/2.jpg",
      "id": 1,
      "height": 655,
      "width": 365
    },

 ...

# The "categories" field contains all category information. If you want to add more detection categories, please add them here. The example is as follows.
  "categories": [
    {
      "supercategory": "cigarette",
      "id": 1,
      "name": "cigarette"
    },
    {
      "supercategory": "Class_Defined_by_Yourself",
      "id": 2,
      "name": "Class_Defined_by_Yourself"
    },

  ...

# The "annotations" field contains information about all instances, including category, bounding box coordinates, id, image id and other information
  "annotations": [
    {
      "category_id": 1,  # Corresponding to the defined category, where 1 represents cigarette
      "bbox": [
        97.0181345931,
        332.7033243081,
        7.5943999555,
        16.4545332369
      ],
      "id": 0,           # Here id is the id serial number of the instance, do not duplicate
      "image_id": 0,     # Here is the id serial number of the image where the instance is located, which may be duplicated. In this case, there are multiple instance objects on one image.
      "iscrowd": 0,
      "area": 124.96230648208665
    },
    {
      "category_id": 2, # Corresponding to the defined category, where 2 represents Class_Defined_by_Yourself
      "bbox": [
        114.3895698372,
        221.9131122343,
        25.9530363697,
        50.5401234568
      ],
      "id": 1,
      "image_id": 1,
      "iscrowd": 0,
      "area": 1311.6696622034585
```

### Configuration File Settings
Refer to [Configuration File](../../../../configs/pphuman/ppyoloe_crn_s_80e_smoking_visdrone.yml), the key should be paid attention to are as follows:
```yaml
metric: COCO
num_classes: 1 # If more categories are added, please modify here accordingly

# Set image_dir，anno_path，dataset_dir correctly
# Ensure that dataset_dir + anno_path can correctly access to the path of the annotation file
# Ensure that dataset_dir + image_dir + the image path in the annotation file can correctly access to the image path
TrainDataset:
  !COCODataSet
    image_dir: ""
    anno_path: smoking_train_cocoformat.json
    dataset_dir: dataset/smoking
    data_fields: ['image', 'gt_bbox', 'gt_class', 'is_crowd']

EvalDataset:
  !COCODataSet
    image_dir: ""
    anno_path: smoking_test_cocoformat.json
    dataset_dir: dataset/smoking

TestDataset:
  !ImageFolder
    anno_path: smoking_test_cocoformat.json
    dataset_dir: dataset/smoking
```

### Model Training And Evaluation
#### Model Training
As [PP-YOLOE](../../../../configs/ppyoloe/README.md), start training with the following command:
```bash
# At Root of PaddleDetection

python -m paddle.distributed.launch --gpus 0,1,2,3  tools/train.py -c configs/pphuman/ppyoloe_crn_s_80e_smoking_visdrone.yml --eval
```

#### Model Evaluation
After training the model, use the following command to evaluate the model metrics.

```bash
# At Root of PaddleDetection

python tools/eval.py -c configs/pphuman/ppyoloe_crn_s_80e_smoking_visdrone.yml
```

#### Model Export
Note: If predicting in Tensor-RT environment, please enable `-o trt=True` for better performance.
```bash
# At Root of PaddleDetection

python tools/export_model.py -c configs/pphuman/ppyoloe_crn_s_80e_smoking_visdrone.yml -o weights=output/ppyoloe_crn_s_80e_smoking_visdrone/best_model trt=True
```

After exporting the model, you can get:
```
ppyoloe_crn_s_80e_smoking_visdrone/
├── infer_cfg.yml
├── model.pdiparams
├── model.pdiparams.info
└── model.pdmodel
```

At this point, this model can be used in PP-Human.

### Custom Action Output
In the model of action recognition based on detection with human id, the task is defined to detect target objects in images of corresponding person. When the target object is detected, the behavior type of the character in a certain period of time. The type of the corresponding classification is regarded as the action of the current period. Therefore, on the basis of completing the training and deployment of the custom model, it is also necessary to convert the detection model results to the final action recognition results as output, and the displayed result of the visualization should be modified.

#### Convert to Action Recognition Result
Please modify the [postprocessing function](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.6/deploy/pipeline/pphuman/action_infer.py#L338).

The core code are:
```python
# Parse the detection model output and filter out valid detection boxes with confidence higher than a threshold.
# Current now,  class 0 is positive, class 1 is negative.
action_ret = {'class': 1.0, 'score': -1.0}
box_num = np_boxes_num[idx]
boxes = det_result['boxes'][cur_box_idx:cur_box_idx + box_num]
cur_box_idx += box_num
isvalid = (boxes[:, 1] > self.threshold) & (boxes[:, 0] == 0)
valid_boxes = boxes[isvalid, :]

if valid_boxes.shape[0] >= 1:
    # When there is a valid detection frame, the category and score of the behavior recognition result are modified accordingly.
    action_ret['class'] = valid_boxes[0, 0]
    action_ret['score'] = valid_boxes[0, 1]
    # Due to the continuity of the action, valid detection results can be reused for a certain number of frames.
    self.result_history[
        tracker_id] = [0, self.frame_life, valid_boxes[0, 1]]
else:
    # If there is no valid detection frame, the result of the current frame is determined according to the historical detection result.
    ...
```

#### Modify Visual Output
At present, ID-based action recognition is displayed based on the results of action recognition and predefined category names. For the detail, please refer to [here](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.6/deploy/pipeline/pipeline.py#L1024-L1043). If the custom action needs to be modified to another display name, please modify it accordingly to output the corresponding result.
