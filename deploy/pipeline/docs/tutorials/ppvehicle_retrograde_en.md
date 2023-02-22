English | [简体中文](ppvehicle_retrograde.md)

# PP-Vehicle vehicle retrograde identification module

Vehicle reverse identification is widely used in smart cities, smart transportation and other directions. In PP-Vehicle, a vehicle retrograde identification module is integrated to identify whether the vehicle is retrograde.

| task | algorithm | precision | infer speed | download|
|-----------|------|-----------|----------|---------------|
| Vehicle detection/tracking | PP-YOLOE | mAP 63.9 | 38.67ms | [infer deploy model](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_ppvehicle.zip) |
| Lane line segmentation | PP-liteseg | mIou 32.69 | 47 ms | [infer deploy model](https://bj.bcebos.com/v1/paddledet/models/pipeline/pp_lite_stdc2_bdd100k.zip) |


Notes：
1. The prediction speed of vehicle detection/tracking model is based on NVIDIA T4 and TensorRT FP16. The model prediction speed includes data preprocessing, model prediction and post-processing.
2. The training and precision test of vehicle detection/tracking model are based on [VeRi](https://www.v7labs.com/open-datasets/veri-dataset).
3. The predicted speed of lane line segmentation model is based on Tesla P40 and python prediction. The predicted speed of the model includes data preprocessing, model prediction and post-processing.
4. Lane line model training and precision testing are based on [BDD100K-LaneSeg](https://bdd-data.berkeley.edu/portal.html#download) and [Apollo Scape](http://apolloscape.auto/lane_segmentation.html#to_dataset_href),The label data of the two data sets is in [Lane_dataset_label](https://bj.bcebos.com/v1/paddledet/data/mot/bdd100k/lane_dataset_label.zip)



## Instructions

### Description of Configuration

[The parameters related to vehicle retrograde in [config file](../../config/infer_cfg_ppvehicle.yml) is as follows:
```
LANE_SEG:
  lane_seg_config: deploy/pipeline/config/lane_seg_config.yml #lane line seg config file
  model_dir: https://bj.bcebos.com/v1/paddledet/models/pipeline/pp_lite_stdc2_bdd100k.zip   #model path

VEHICLE_RETROGRADE:
  frame_len: 8                        #Number of sampling frames
  sample_freq: 7                      #sampling frequency
  enable: True                        #Whether to enable the funcion
  filter_horizontal_flag: False       #Whether to filter vehicles in horizontal direction
  keep_right_flag: True               #According to the right driving rule, if the vehicle keeps left driving, it is set as False
  deviation: 23                       #Filter the horizontal angle vehicles threshold. If it is greater than this angle, filter
  move_scale: 0.01                    #Filter the threshold value of stationary vehicles. If the vehicle moving pixel is greater than the image diagonal * move_scale, the vehicle is considered moving, otherwise, the vehicle is stationary
  fence_line: []                      #Lane centerline coordinates, format[x1,y1,x2,y2] and y2>y1. If it is empty, the program will automatically judge according to the direction of traffic flow
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
2. Modify Profile`VEHICLE_RETROGRADE`-`enable: True`， item to enable this function.



3. When video input is required for vehicle retrograde recognition function, the starting command is as follows:

```bash
#For single video
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_ppvehicle.yml \
                                   -o VEHICLE_RETROGRADE.enable=true \
                                   --video_file=test_video.mp4 \
                                   --device=gpu

#For folder contains one or multiple videos
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_ppvehicle.yml \
                                   -o VEHICLE_RETROGRADE.enable=true \
                                   --video_dir=test_video \
                                   --device=gpu
```

5. There are two ways to modify the model path:

    - 1.Set paths of each model in `./deploy/pipeline/config/infer_cfg_ppvehicle.yml`,For Lane line segmentation, the path should be modified under the `LANE_SEG`
    - 2.Directly add `-o` in command line to override the default model path in the configuration file:

```bash
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_ppvehicle.yml \
                                   --video_file=test_video.mp4 \
                                   --device=gpu \
                                   -o LANE_SEG.model_dir=output_inference/
                                   VEHICLE_RETROGRADE.enable=true

```
The result is shown as follow:

<div width="1000" align="center">
  <img src="https://raw.githubusercontent.com/LokeZhou/PaddleDetection/develop/deploy/pipeline/docs/images/vehicle_retrograde.gif"/>
</div>

**Note:**
 - Automatic judgment condition of lane line middle line: there are two vehicles in opposite directions in the sampled video segment, and the judgment is fixed after one time and will not be updated;
 - Due to camera angle and 2d visual angle problems, the judgment of lane line middle line is inaccurate.
 - You can manually enter the middle line coordinates in the configuration file.Example as [infer_cfg_vehicle_violation.yml](../../config/examples/infer_cfg_vehicle_violation.yml)


## Features to the Solution
1.In the sampling video segment, judge whether the vehicle is retrograde according to the location of the lane centerline and the vehicle track, and determine the flow chart:
<div width="1000" align="center">
  <img src="https://raw.githubusercontent.com/LokeZhou/PaddleDetection/develop/deploy/pipeline/docs/images/vehicle_retrograde_en.png"/>
</div>

2.Lane line recognition model uses [PaddleSeg]（ https://github.com/PaddlePaddle/PaddleSeg ）Super lightweight segmentation scheme.Train [lable](https://bj.bcebos.com/v1/paddledet/data/mot/bdd100k/lane_dataset_label.zip)it is divided into four categories:
  0 Background
  1 Double yellow line
  2 Solid line
  3 Dashed line
Lane line recognition filtering Dashed lines;

3.Lane lines are obtained by clustering segmentation results, and the horizontal lane lines are filtered by default. If not, you can modify the `filter_flag` in [lane line seg config file](../../config/lane_seg_config.yml);

4.The vehicles in the horizontal direction are filtered by default when judging the vehicles in the reverse direction. If not, you can modify the `filter_horizontal_flag` in [config file](../../config/infer_cfg_ppvehicle.yml);

5.The vehicle will be judged according to the right driving rule by default.If not, you can modify the `keep_right_flag` in [config file](../../config/infer_cfg_ppvehicle.yml);


**Performance optimization measures：**
1.Due to the camera's viewing angle, it can be decided whether to filter the lane lines and vehicles in the horizontal direction according to the actual situation;

2.The lane middle line can be manually entered;
