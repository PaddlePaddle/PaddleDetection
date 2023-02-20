
# PP-Vehicle Illegal Parking Recognition Module

Illegal parking recognition in no-parking areas has a very wide range of applications in vehicle application scenarios. With the help of AI, human input can be reduced, and illegally parked vehicles can be accurately and quickly identified, and further behaviors such as broadcasting to expel the vehicles can be performed. Based on the vehicle tracking model, license plate detection model and license plate recognition model, the PP-Vehicle realizes the illegal parking recognition function. The specific model information is as follows:

| Task                 | Algorithm | Precision | Inference Speed(ms) |Inference Model Download Link                                                                               |
|:---------------------|:---------:|:------:|:------:| :---------------------------------------------------------------------------------: |
| Vehicle Tracking |  PP-YOLOE-l | mAP: 63.9 | - |[Link](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_ppvehicle.zip) |
| Plate Detection |  ch_PP-OCRv3_det  |  hmean: 0.979  | - | [Link](https://bj.bcebos.com/v1/paddledet/models/pipeline/ch_PP-OCRv3_det_infer.tar.gz) |
| Plate Recognition    |  ch_PP-OCRv3_rec  |  acc: 0.773  | - | [Link](https://bj.bcebos.com/v1/paddledet/models/pipeline/ch_PP-OCRv3_rec_infer.tar.gz) |

1. The tracking model uses the PPVehicle dataset (integrating BDD100K-MOT and UA-DETRAC), which combines car, truck, bus, van in BDD100K-MOT and car, bus, and van in UA-DETRAC into one class which named vehicle (1).
2. The license plate detection and recognition model is fine-tuned on the CCPD2019 and CCPD2020 using the PP-OCRv3 model.

## Instructions

1. Users can download the model from the link in the table above and unzip it to the ``PaddleDetection/output_inference``` path, and modify the model path in the configuration file, or download the model automatically by default. The model paths for the three models can be manually set in ``deploy/pipeline/config/examples/infer_cfg_illegal_parking.yml```.

Description of configuration items in `infer_cfg_illegal_parking.yml`:
```
MOT:                                                                                             # Tracking Module
  model_dir: https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_ppvehicle.zip  # Path of Tracking Model
  tracker_config: deploy/pipeline/config/tracker_config.yml                                      # Config Path of Tracking
  batch_size: 1                                                                                  # Tracking batch size
  enable: True                                                                                   # Whether to Enable Tracking Function

VEHICLE_PLATE:                                                                                   # Plate Recognition Module
  det_model_dir: https://bj.bcebos.com/v1/paddledet/models/pipeline/ch_PP-OCRv3_det_infer.tar.gz # Path of Plate Detection Model
  det_limit_side_len: 480                                                                        # Single Side Size of Detection Model
  det_limit_type: "max"                                                                          # Detection model Input Size Selection of Long and Short Sides, "max" Represents the Long Side
  rec_model_dir: https://bj.bcebos.com/v1/paddledet/models/pipeline/ch_PP-OCRv3_rec_infer.tar.gz # Path of Plate Recognition Model
  rec_image_shape: [3, 48, 320]                                                                  # The Input Size of Plate Recognition Model
  rec_batch_num: 6                                                                               # Plate Recognition batch size
  word_dict_path: deploy/pipeline/ppvehicle/rec_word_dict.txt                                    # OCR Model Look-up Table
  enable: True                                                                                   # Whether to Enable Plate Recognition Function
```

2. Input video, the command is as follows:
```python
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/examples/infer_cfg_illegal_parking.yml \
                                                   --video_file=test_video.mp4 \
                                                   --device=gpu \
                                                   --draw_center_traj \
                                                   --illegal_parking_time=5 \
                                                   --region_type=custom \
                                                   --region_polygon 100 1000 1000 1000 900 1700 0 1700

The parameter description:
- config: config path;
- video_file: video path to be tested;
- device: device to infe;
- draw_center_traj: draw the trajectory of the center of the vehicle;
- illegal_parking_time: illegal parking time, in seconds;
- region_type: illegal parking region type, 'custom' means the region is customized;
- region_polygon: customized illegal parking region which includes three points at least.

3. Methods to modify the path of model:

    - Method 1: Configure different model paths in ```./deploy/pipeline/config/examples/infer_cfg_illegal_parking.yml``` file;
    - Method2: In the command line, add `-o VEHICLE_PLATE.det_model_dir=[YOUR_DETMODEL_PATH] VEHICLE_PLATE.rec_model_dir=[YOUR_RECMODEL_PATH]` after the --config configuration item to modify the model path.


Test Result:

<div width="600" align="center">
  <img src="https://user-images.githubusercontent.com/22989727/205598624-bcf5165c-990c-4fe4-8cde-eb1d45298d8f.gif"/>
</div>


## Method Description

1. Target multi-target tracking obtains the vehicle detection frame in the picture/video input. The model scheme is PP-YOLOE. For detailed documentation, refer to [PP-YOLOE](../../../configs/ppyoloe/README_cn. md)
2. Obtain the trajectory of each vehicle based on the tracking algorithm. If the center of the vehicle is in the illegal parking area and does not move within the specified time, it is considered illegal parking;
3. Use the license plate recognition model to get the illegal parking license plate and visualize it.


## References

1. Detection Model in PaddeDetection:[PP-YOLOE](../../../../configs/ppyoloe).
2. Character Recognition Model Library in Paddle: [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR).
