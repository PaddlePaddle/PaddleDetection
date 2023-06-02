English | [简体中文](ppvehicle_press.md)

# PP-Vehicle press line identification module

Vehicle compaction line recognition is widely used in smart cities, smart transportation and other directions.
In PP-Vehicle, a vehicle compaction line identification module is integrated to identify whether the vehicle is in violation of regulations.

| task | algorithm | precision | infer speed | download|
|-----------|------|-----------|----------|---------------|
| Vehicle detection/tracking | PP-YOLOE | mAP 63.9 | 38.67ms | [infer deploy model](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_ppvehicle.zip) |
| Lane line segmentation | PP-liteseg | mIou 32.69 | 47 ms | [infer deploy model](https://bj.bcebos.com/v1/paddledet/models/pipeline/pp_lite_stdc2_bdd100k.zip) |


Notes：
1. The prediction speed of vehicle detection/tracking model is based on NVIDIA T4 and TensorRT FP16. The model prediction speed includes data preprocessing, model prediction and post-processing.
2. The training and precision test of vehicle detection/tracking model are based on [VeRi](https://www.v7labs.com/open-datasets/veri-dataset).
3. The predicted speed of lane line segmentation model is based on Tesla P40 and python prediction. The predicted speed of the model includes data preprocessing, model prediction and post-processing.
4. Lane line model training and precision testing are based on [BDD100K-LaneSeg](https://bdd-data.berkeley.edu/portal.html#download)and [Apollo Scape](http://apolloscape.auto/lane_segmentation.html#to_dataset_href),The label data of the two data sets is in[Lane_dataset_label](https://bj.bcebos.com/v1/paddledet/data/mot/bdd100k/lane_dataset_label.zip)


## Instructions

### Description of Configuration

The parameters related to vehicle line pressing in [config file](../../config/infer_cfg_ppvehicle.yml) is as follows:
```
VEHICLE_PRESSING:
  enable: True               #Whether to enable the funcion
LANE_SEG:
  lane_seg_config: deploy/pipeline/config/lane_seg_config.yml #lane line seg config file
  model_dir: https://bj.bcebos.com/v1/paddledet/models/pipeline/pp_lite_stdc2_bdd100k.zip   #model path
```
The parameters related to Lane line segmentation in [lane line seg config file](../../config/lane_seg_config.yml)is as follows:
```
type: PLSLaneseg  #Select segmentation Model

PLSLaneseg:
  batch_size: 1                                       #image batch_size
  device: gpu                                         #device is gpu or cpu
  filter_flag: True                                   #Whether to filter the horizontal direction road route
  horizontal_filtration_degree: 23                    #Filter the threshold value of the lane line in the horizontal direction. When the difference between the maximum inclination angle and the minimum inclination angle of the segmented lane line is less than the threshold value, no filtering is performed

  horizontal_filtering_threshold: 0.25                #Determine the threshold value for separating the vertical direction from the horizontal direction thr=(min_degree+max_degree) * 0.25 Divide the lane line into vertical direction and horizontal direction according to the comparison between the gradient angle of the lane line and thr
```
### How to Use

1. Download 'vehicle detection/tracking' and 'lane line recognition' two prediction deployment models from the model base and unzip them to '/ output_ Invitation ` under the path; By default, the model will be downloaded automatically. If you download it manually, you need to modify the model folder as the model storage path.

2. Modify Profile ` VEHICLE_PRESSING ' -'enable: True'  item to enable this function.

3. When inputting a picture, the startup command is as follows (for more command parameter descriptions,please refer to [QUICK_STARTED - Parameter_Description](./PPVehicle_QUICK_STARTED.md)

```bash
# For single image
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_ppvehicle.yml \
                                   -o VEHICLE_PRESSING.enable=true
                                   --image_file=test_image.jpg \
                                   --device=gpu

# For folder contains one or multiple images
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_ppvehicle.yml \
                                   -o VEHICLE_PRESSING.enable=true
                                   --image_dir=images/ \
                                   --device=gpu
```

4. For video input, please run these commands.

```bash
#For single video
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_ppvehicle.yml \
                                   -o VEHICLE_PRESSING.enable=true
                                   --video_file=test_video.mp4 \
                                   --device=gpu

#For folder contains one or multiple videos
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_ppvehicle.yml \
                                   --video_dir=test_videos/ \
                                   -o VEHICLE_PRESSING.enable=true
                                   --device=gpu
```

5. There are two ways to modify the model path:

    - 1.Set paths of each model in `./deploy/pipeline/config/infer_cfg_ppvehicle.yml`,For Lane line segmentation, the path should be modified under the `LANE_SEG`
    - 2.Directly add `-o` in command line to override the default model path in the configuration file:

```bash
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_ppvehicle.yml \
                                                   --video_file=test_video.mp4 \
                                                   --device=gpu \
                                                   -o VEHICLE_PRESSING.enable=true
                                                   LANE_SEG.model_dir=output_inference
```

The result is shown as follow:

<div width="1000" align="center">
  <img src="https://raw.githubusercontent.com/LokeZhou/PaddleDetection/develop/deploy/pipeline/docs/images/vehicle_press.gif"/>
</div>

## Features to the Solution
1.Lane line recognition model uses [PaddleSeg]（ https://github.com/PaddlePaddle/PaddleSeg ）Super lightweight segmentation scheme.Train [lable](https://bj.bcebos.com/v1/paddledet/data/mot/bdd100k/lane_dataset_label.zip)it is divided into four categories:
  0 Background
  1 Double yellow line
  2 Solid line
  3 Dashed line
Lane line recognition filtering Dashed lines;

2.Lane lines are obtained by clustering segmentation results, and the horizontal lane lines are filtered by default. If not, you can modify the `filter_flag` in [lane line seg config file](../../config/lane_seg_config.yml);

3.Judgment conditions for vehicle line pressing: whether there is intersection between the bottom edge line of vehicle detection frame and lane line;

**Performance optimization measures：**
1.Due to the camera angle, it can be decided whether to filter the lane line in the horizontal direction according to the actual situation;
