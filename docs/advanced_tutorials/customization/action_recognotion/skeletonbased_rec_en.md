[简体中文](./skeletonbased_rec.md) | English

# Skeleton-based action recognition

## Environmental Preparation
The skeleton-based action recognition is trained with [PaddleVideo](https://github.com/PaddlePaddle/PaddleVideo). Please refer to [Installation](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/en/install.md) to complete the environment installation for subsequent model training and usage processes.

## Data Preparation
For the model of skeleton-based model, you can refer to [this document](https://github.com/PaddlePaddle/PaddleVideo/tree/develop/applications/PPHuman#%E5%87%86%E5%A4%87%E8%AE %AD%E7%BB%83%E6%95%B0%E6%8D%AE) to preparation training adapted to PaddleVideo. The main process includes the following steps:


### Data Format Description
STGCN is a model based on the sequence of skeleton point coordinates. In PaddleVideo, training data is `Numpy` data stored with `.npy` format, and labels can be files stored in `.npy` or `.pkl` format. The dimension requirement for sequence data is `(N,C,T,V,M)`, the current solution only supports behaviors composed of a single person (but there can be multiple people in the video, and each person performs action recognition separately), that is` M=1`.

| Dim | Size | Description |
| ---- | ---- | ---------- |
| N | Not Fixed | The number of sequences in the dataset |
| C | 2 | Keypoint coordinate, i.e. (x, y) |
| T | 50 | The temporal dimension of the action sequence (i.e. the number of continuous frames)|
| V | 17 | The number of keypoints of each person, here we use the definition of the `COCO` dataset, see [here](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/docs/tutorials/PrepareKeypointDataSet_en.md#description-for-coco-datasetkeypoint) |
| M | 1 | The number of persons, here we only predict a single person for each action sequence |

### Get The Skeleton Point Coordinates of The Sequence
For a sequence to be labeled (here a sequence refers to an action segment, which can be a video or an ordered collection of pictures). The coordinates of skeletal points (also known as keypoints) can be obtained through model prediction or manual annotation.
- Model prediction: You can directly select the model in the [PaddleDetection KeyPoint Models](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/configs/keypoint/README_en.md) and according to `3, training and testing - Deployment Prediction - Detect + keypoint top-down model joint deployment` to get the 17 keypoint coordinates of the target sequence.

When using the model to predict and obtain the coordinates, you can refer to the following steps, please note that the operation in PaddleDetection at this time.

```bash
# current path is under root of PaddleDetection

# Step 1: download pretrained inference models.
wget https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip
wget https://bj.bcebos.com/v1/paddledet/models/pipeline/dark_hrnet_w32_256x192.zip
unzip -d output_inference/ mot_ppyoloe_l_36e_pipeline.zip
unzip -d output_inference/ dark_hrnet_w32_256x192.zip

# Step 2: Get the keypoint coordinarys

# if your data is image sequence
python deploy/python/det_keypoint_unite_infer.py --det_model_dir=output_inference/mot_ppyoloe_l_36e_pipeline/ --keypoint_model_dir=output_inference/dark_hrnet_w32_256x192 --image_dir={your image directory path} --device=GPU --save_res=True

# if your data is video
python deploy/python/det_keypoint_unite_infer.py --det_model_dir=output_inference/mot_ppyoloe_l_36e_pipeline/ --keypoint_model_dir=output_inference/dark_hrnet_w32_256x192 --video_file={your video file path} --device=GPU --save_res=True
```
We can get a detection result file named `det_keypoint_unite_image_results.json`. The detail of content can be seen at [Here](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/deploy/python/det_keypoint_unite_infer.py#L108).


### Uniform Sequence Length
Since the length of each action in the actual data is different, the first step is to pre-determine the time sequence length according to your data and the actual scene (in PP-Human, we use 50 frames as an action sequence), and do the following processing to the data:
- If the actual length exceeds the predetermined length, a 50-frame segment will be randomly intercepted
- Data whose actual length is less than the predetermined length: fill with 0 until 50 frames are met
- data exactly equal to the predeter: no processing required

Note: After this step is completed, please strictly confirm that the processed data contains a complete action, and there will be no ambiguity in prediction. It is recommended to confirm by visualizing the data.

### Save to PaddleVideo usable formats
After the first two steps of processing, we get the annotation of each character action fragment. At this time, we have a list `all_kpts`, which contains multiple keypoint sequence fragments, each one has a shape of (T, V, C) (in our case (50, 17, 2)), which is further converted into a format usable by PaddleVideo.
- Adjust dimension order: `np.transpose` and `np.expand_dims` can be used to convert the dimension of each fragment into (C, T, V, M) format.
- Combine and save all clips as one file

Note: `class_id` is a `int` type variable, similar to other classification tasks. For example `0: falling, 1: other`.

We provide a [script file](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/applications/PPHuman/datasets/prepare_dataset.py) to do this step, which can directly process the generated `det_keypoint_unite_image_results.json` file. The content executed by the script includes parsing the content of the json file, unforming the training data  sequence and saving the data file as described in the preceding steps.

```bash
mkdir {root of PaddleVideo}/applications/PPHuman/datasets/annotations

mv det_keypoint_unite_image_results.json {root of PaddleVideo}/applications/PPHuman/datasets/annotations/det_keypoint_unite_image_results_{video_id}_{camera_id}.json

cd {root of PaddleVideo}/applications/PPHuman/datasets/

python prepare_dataset.py
```

Now, we have available training data (`.npy`) and corresponding annotation files (`.pkl`).

## Model Optimization
### detection-tracking model optimization
The performance of action recognition based on skelenton depends on the pre-order detection and tracking models. If the pedestrian location cannot be accurately detected in the actual scene, or it is difficult to correctly assign the person ID between different frames, the performance of the action recognition part will be limited. If you encounter the above problems in actual use, please refer to [Secondary Development of Detection Task](../detection_en.md) and [Secondary Development of Multi-target Tracking Task](../pphuman_mot_en.md) for detection/track model optimization.

### keypoint model optimization
As the core feature of the scheme, the skeleton point positioning performance also determines the overall effect of action recognition. If there are obvious errors in the recognition results of the keypoint coordinates of in the actual scene, it is difficult to distinguish the specific actions from the skeleton image composed of the keypoint.
You can refer to [Secondary Development of Keypoint Detection Task](../keypoint_detection_en.md) to optimize the keypoint model.

### Coordinate Normalization
After getting coordinates of the skeleton points, it is recommended to perform normalization processing according to the detection bounding box of each person to reduce the convergence difficulty brought by the difference in the position and scale of the person.

## Add New Action

In skeleton-based action recognition, the model is [ST-GCN](https://arxiv.org/abs/1801.07455). Modified to adapt PaddleVideo based on [Training Step](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/en/model_zoo/recognition/stgcn.md). And complete the model training and exporting process.

### Data Preparation And Configuration File Settings
- Prepare the training data (`.npy`) and the corresponding annotation file (`.pkl`) according to `Data preparation`. Correspondingly placed under `{root of PaddleVideo}/applications/PPHuman/datasets/`.

- Refer [Configuration File](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/applications/PPHuman/configs/stgcn_pphuman.yaml), the things to focus on are as follows:

```yaml
MODEL: #MODEL field
    framework:
        backbone:
        name: "STGCN"
        in_channels: 2  # This corresponds to the C dimension in the data format description, representing two-dimensional coordinates.
        dropout: 0.5
        layout: 'coco_keypoint'
        data_bn: True
    head:
        name: "STGCNHead"
        num_classes: 2  # If there are multiple action types in the data, this needs to be modified to match the number of types.
    if_top5: False # When the number of action types is less than 5, please set it to False, otherwise an error will be raised.

...


# Please set the data and label path of the train/valid/test part correctly according to the data path
DATASET: #DATASET field
    batch_size: 64
    num_workers: 4
    test_batch_size: 1
    test_num_workers: 0
    train:
        format: "SkeletonDataset" #Mandatory, indicate the type of dataset, associate to the 'paddle
        file_path: "./applications/PPHuman/datasets/train_data.npy" #mandatory, train data index file path
        label_path: "./applications/PPHuman/datasets/train_label.pkl"

    valid:
        format: "SkeletonDataset" #Mandatory, indicate the type of dataset, associate to the 'paddlevideo/loader/dateset'
        file_path: "./applications/PPHuman/datasets/val_data.npy" #Mandatory, valid data index file path
        label_path: "./applications/PPHuman/datasets/val_label.pkl"

        test_mode: True
    test:
        format: "SkeletonDataset" #Mandatory, indicate the type of dataset, associate to the 'paddlevideo/loader/dateset'
        file_path: "./applications/PPHuman/datasets/val_data.npy" #Mandatory, valid data index file path
        label_path: "./applications/PPHuman/datasets/val_label.pkl"

        test_mode: True
```

### Model Training And Evaluation

- In PaddleVideo, start training with the following command:
```bash
# current path is under root of PaddleVideo
python main.py -c applications/PPHuman/configs/stgcn_pphuman.yaml

# Since the task may overfit, it is recommended to evaluate model during training to save the best model.
python main.py --validate -c applications/PPHuman/configs/stgcn_pphuman.yaml
```

- After training the model, use the following command to do inference.
```bash
python main.py --test -c applications/PPHuman/configs/stgcn_pphuman.yaml  -w output/STGCN/STGCN_best.pdparams
```

### Model Export

In PaddleVideo, use the following command to export model and get structure file `STGCN.pdmodel` and weight file `STGCN.pdiparams`. And add the configuration file here.
```bash
# current path is under root of PaddleVideo
python tools/export_model.py -c applications/PPHuman/configs/stgcn_pphuman.yaml \
                                -p output/STGCN/STGCN_best.pdparams \
                                -o output_inference/STGCN

cp applications/PPHuman/configs/infer_cfg.yml output_inference/STGCN

# Rename model files to adapt PP-Human
cd output_inference/STGCN
mv STGCN.pdiparams model.pdiparams
mv STGCN.pdiparams.info model.pdiparams.info
mv STGCN.pdmodel model.pdmodel
```

The directory structure will look like:
```
STGCN
├── infer_cfg.yml
├── model.pdiparams
├── model.pdiparams.info
├── model.pdmodel
```
At this point, this model can be used in PP-Human.

**Note**: If the length of the video sequence or the number of keypoints is changed during training, the content of the `INFERENCE` field in the configuration file needs to be modified accordingly to correct prediction.

```yaml
# The dimension of the sequence data is (N,C,T,V,M)
INFERENCE:
    name: 'STGCN_Inference_helper'
    num_channels: 2 # Corresponding to C dimension
    window_size: 50 # Corresponding to T dimension, please set it accordingly to the sequence length.
    vertex_nums: 17 # Corresponding to V dimension, please set it accordingly to the number of keypoints
    person_nums: 1 # Corresponding to M dimension
```

### Custom Action Output
In the skeleton-based action recognition, the classification result of the model represents the behavior type of the character in a certain period of time. The type of the corresponding classification is regarded as the action of the current period. Therefore, on the basis of completing the training and deployment of the custom model, the model output is directly used as the final result, and the displayed result of the visualization should be modified.

#### Modify Visual Output
At present, ID-based action recognition is displayed based on the results of action recognition and predefined category names. For the detail, please refer to [here](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.6/deploy/pipeline/pipeline.py#L1024-L1043). If the custom action needs to be modified to another display name, please modify it accordingly to output the corresponding result.
