English | [简体中文](ppvehicle_mot.md)

# PP-Vehicle Vehicle Tracking Module

【Application Introduction】

Vehicle detection and tracking are widely used in traffic monitoring and autonomous driving. The detection and tracking module is integrated in PP-Vehicle, providing a solid foundation for tasks including license plate detection and vehicle attribute recognition. We provide pre-trained models that can be directly used by developers.

【Model Download】

| Task                       | Algorithm  | Accuracy                | Inference speed(ms)                  | Download Link                                                                              |
| -------------------------- | ---------- | ----------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------ |
| Vehicle Detection/Tracking | PP-YOLOE-l | mAP: 63.9<br>MOTA: 50.1 | Detection: 25.1ms<br>Tracking：31.8ms | [Link](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_ppvehicle.zip) |
| Vehicle Detection/Tracking | PP-YOLOE-s | mAP: 61.3<br>MOTA: 46.8 | Detection: 16.2ms<br>Tracking：21.0ms | [Link](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_s_36e_ppvehicle.zip) |

1. The detection/tracking model uses the PPVehicle dataset ( which integrates BDD100K-MOT and UA-DETRAC). The dataset merged car, truck, bus, van from BDD100K-MOT and car, bus, van from UA-DETRAC all into 1 class vehicle(1). The detection accuracy mAP was tested on the test set of PPVehicle, and the tracking accuracy MOTA was obtained on the test set of BDD100K-MOT (`car, truck, bus, van` were combined into 1 class `vehicle`). For more details about the training procedure, please refer to [ppvehicle](../../../../configs/ppvehicle).
2. Inference speed is obtained at T4 with TensorRT FP16 enabled, which includes data pre-processing, model inference and post-processing.

## How To Use

【Config】

The parameters associated with the attributes in the configuration file are as follows.

```
DET:
  model_dir: output_inference/mot_ppyoloe_l_36e_ppvehicle/ # Vehicle detection model path
  batch_size: 1   # Batch_size size for model inference

MOT:
  model_dir: output_inference/mot_ppyoloe_l_36e_ppvehicle/ # Vehicle tracking model path
  tracker_config: deploy/pipeline/config/tracker_config.yml
  batch_size: 1   # Batch_size size for model inference, 1 only for tracking task.
  skip_frame_num: -1  # Number of frames to skip, -1 means no skipping, the maximum skipped frames are recommended to be 3
  enable: False   # Whether or not to enable this function,  please make sure it is set to True before tracking
```

【Usage】

1. Download the model from the link in the table above and unzip it to ``. /output_inference`` and change the model path in the configuration file. The default is to download the model automatically, no changes are needed.

2. The image input will start a pure detection task, and the start command is as follows

   ```python
   python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_ppvehicle.yml \
                                                   --image_file=test_image.jpg \
                                                   --device=gpu
   ```

3. Video input will start a tracking task. Please set `enable=True` for the MOT configuration in infer_cfg_ppvehicle.yml. If skip frames are needed for faster detection and tracking,  it is recommended to set `skip_frame_num: 2`  the maximum should not exceed 3.

   ```
   MOT:
   model_dir: https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_ppvehicle.zip
   tracker_config: deploy/pipeline/config/tracker_config.yml
   batch_size: 1
   skip_frame_num: 2
   enable: True
   ```

   ```python
   python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_ppvehicle.yml \
                                                   --video_file=test_video.mp4 \
                                                   --device=gpu
   ```

4. There are two ways to modify the model path

   - Config different model path in```./deploy/pipeline/config/infer_cfg_ppvehicle.yml```. The detection and tracking models correspond to the `DET` and `MOT` fields respectively. Modify the path under the corresponding field to the actual path.

   - **[Recommand]** Add`-o MOT.model_dir=[YOUR_DETMODEL_PATH]` after the config in the command line to modify model path

     ```python
     python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_ppvehicle.yml \
                                                  --video_file=test_video.mp4 \
                                                  --device=gpu \
                                                  --region_type=horizontal \
                                                  --do_entrance_counting \
                                                  --draw_center_traj \
                                                  -o MOT.model_dir=ppyoloe/
     ```

**Note:**

- `--do_entrance_counting` : Whether to count entrance/exit traffic flows, the default is False

- `--draw_center_traj` : Whether to draw center trajectory, the default is False. Its input video is preferably taken from a still camera

- `--region_type` : The region for traffic counting. When setting `--do_entrance_counting`, there are two options: `horizontal` or `vertical`. The default is `horizontal`, which means the center horizontal line of the video picture is the entrance and exit. When the center point of the same object frame is on both sides of the centre horizontal line of the region in two adjacent seconds, i.e. the count adds 1.
5. Regional break-in and counting

Please set the MOT config: enable=True in `infer_cfg_ppvehicle.yml` before running the starting command:

```
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_ppvehicle.yml \
 --video_file=test_video.mp4 \
 --device=gpu \
 --draw_center_traj \
 --do_break_in_counting \
 --region_type=custom \
 --region_polygon 200 200 400 200 300 400 100 400
```

**Note:**

- Test video of area break-ins must be taken from a still camera, with no shaky or moving footage.

- `--do_break_in_counting`Indicates whether or not to count the entrance and exit of the area. The default is False.

- `--region_type` indicates the region for traffic counting, when setting `--do_break_in_counting` only `custom` can be selected, and the default is `custom`. It means that the customized region is used as the entry and exit. When the coordinates of the lower boundary midpoint of the same object frame goes to the inside of the region within two adjacent seconds, i.e. the count adds one.

- `--region_polygon` indicates a sequence of point coordinates for a polygon in a customized region. Every two are a pair of point coordinates (x,y). **In clockwise order** they are connected into a **closed region**, at least 3 pairs of points are needed (or 6 integers). The default value is `[]`. Developers need to set the point coordinates manually. If it is a quadrilateral region, the coordinate order is `top left, top right , bottom right, bottom left`. Developers can run [this code](... /... /tools/get_video_info.py) to get the resolution frames of the predicted video. It also supports customizing and adjusting the visualisation of the polygon area.
  The code for the visualisation of the customized polygon area runs as follows.

  ```python
  python get_video_info.py --video_file=demo.mp4 --region_polygon 200 200 400 200 300 400 100 400
  ```

  A quick tip for drawing customized area: first take any point to get the picture, open it with the drawing tool, mouse over the area point and the coordinates will be displayed, record it and round it up, use it as a region_polygon parameter for this visualisation code and run the visualisation again, and fine-tune the point coordinates parameter.

【Showcase】

<div width="600" align="center">
  <img src="../images/mot_vehicle.gif"/>
</div>

## Solution

【Solution and feature】

- PP-YOLOE is adopted for vehicle detection frame of object detection, multi-object tracking in the picture/video input. For details, please refer to [PP-YOLOE](... /... /... /configs/ppyoloe/README_cn.md) and [PPVehicle](../../../../configs/ppvehicle)
- [OC-SORT](https://arxiv.org/pdf/2203.14360.pdf) is adopted as multi-object tracking model. PP-YOLOE replaced YOLOX as detector, and OCSORTTracker is the tracker. For more details, please refer to [OC-SORT](../../../../configs/mot/ocsort)

## Reference

```
@article{cao2022observation,
  title={Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking},
  author={Cao, Jinkun and Weng, Xinshuo and Khirodkar, Rawal and Pang, Jiangmiao and Kitani, Kris},
  journal={arXiv preprint arXiv:2203.14360},
  year={2022}
}
```
