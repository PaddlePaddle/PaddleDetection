English | [简体中文](pphuman_mot.md)

# Detection and Tracking Module of PP-Human

Pedestrian detection and tracking is widely used in the intelligent community, industrial inspection, transportation monitoring and so on. PP-Human has the detection and tracking module, which is fundamental to keypoint detection, attribute action recognition, etc. Users enjoy easy access to pretrained models here.

| Task                 | Algorithm | Precision | Inference Speed(ms) | Download Link                                                                               |
|:---------------------|:---------:|:------:|:------:| :---------------------------------------------------------------------------------: |
| Pedestrian Detection/ Tracking    |  PP-YOLOE-l | mAP: 57.8 <br> MOTA: 82.2 | Detection: 25.1ms <br> Tracking：31.8ms | [Download](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip) |
| Pedestrian Detection/ Tracking    |  PP-YOLOE-s | mAP: 53.2 <br> MOTA: 73.9 | Detection: 16.2ms <br> Tracking：21.0ms | [Download](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_s_36e_pipeline.zip) |

1. The precision of the pedestrian detection/ tracking model is obtained by trainning and testing on [COCO-Person](http://cocodataset.org/), [CrowdHuman](http://www.crowdhuman.org/), [HIEVE](http://humaninevents.org/) and some business data.
2. The inference speed is the speed of using TensorRT FP16 on T4, the total number of data pre-training, model inference, and post-processing.

## How to Use

1. Download models from the links of the above table and unizp them to ```./output_inference```.
2. When use the image as input, it's a detection task, the start command is as follows:
```python
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_pphuman.yml \
                                                   --image_file=test_image.jpg \
                                                   --device=gpu
```
3. When use the video as input, it's a tracking task, first you should set the "enable: True" in MOT of infer_cfg_pphuman.yml. If you want skip some frames speed up the detection and tracking process, you can set `skip_frame_num: 2`, it is recommended that the maximum number of skip_frame_num should not exceed 3:
```
MOT:
  model_dir: https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip
  tracker_config: deploy/pipeline/config/tracker_config.yml
  batch_size: 1
  skip_frame_num: 2
  enable: True
```
and then the start command is as follows:
```python
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_pphuman.yml \
                                                   --video_file=test_video.mp4 \
                                                   --device=gpu
```
4. There are two ways to modify the model path:

     - In `./deploy/pipeline/config/infer_cfg_pphuman.yml`, you can configurate different model paths，which is proper only if you match keypoint models and action recognition models with the fields of `DET` and `MOT` respectively, and modify the corresponding path of each field into the expected path.
    - Add `-o MOT.model_dir` in the command line following the --config to change the model path：

```python
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_pphuman.yml \
                                                   -o MOT.model_dir=ppyoloe/\
                                                   --video_file=test_video.mp4 \
                                                   --device=gpu \
                                                   --region_type=horizontal \
                                                   --do_entrance_counting \
                                                   --draw_center_traj

```
**Note:**

 - `--do_entrance_counting` is whether to calculate flow at the gateway, and the default setting is False.
 - `--draw_center_traj` means whether to draw the track, and the default setting is False. It's worth noting that the test video of track drawing should be filmed by the still camera.
 - `--region_type` means the region type of flow counting. When set `--do_entrance_counting`, you can select from `horizontal` or `vertical`, the default setting is `horizontal`, means that the central horizontal line of the video picture is used as the entrance and exit, and when the central point of the same object box is on both sides of the central horizontal line of the area in two adjacent seconds, the counting plus one is completed.

The test result is：

<div width="600" align="center">
  <img src="https://user-images.githubusercontent.com/22989727/205599943-8da89ce8-f6d1-47e5-adc8-6d199b76d167.gif"/>
</div>

Data source and copyright owner：Skyinfor Technology. Thanks for the provision of actual scenario data, which are only used for academic research here.

5. Break in and counting

Please set the "enable: True" in MOT of infer_cfg_pphuman.yml at first, and then the start command is as follows:
```python
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_pphuman.yml \
                                                   --video_file=test_video.mp4 \
                                                   --device=gpu \
                                                   --draw_center_traj \
                                                   --do_break_in_counting \
                                                   --region_type=custom \
                                                   --region_polygon 200 200 400 200 300 400 100 400
```

**Note:**
 - `--do_break_in_counting` is whether to calculate flow when break in the user-defined region, and the default setting is False.
 - `--region_type` means the region type of flow counting. When set `--do_break_in_counting`, only `custom` can be selected, and the default is `custom`, which means that the user-defined region is used as the entrance and exit, and when the midpoint coords of the bottom boundary of the same object moves from outside to inside the region within two adjacent seconds, the counting plus one is completed.
 - `--region_polygon` means the point coords sequence of the polygon in the user-defined region. Every two integers are a pair of point coords (x,y), which are connected into a closed area in clockwise order. At least 3 pairs of points, that is, 6 integers, are required. The default value is `[]`, and the user needs to set the point coords by himself. Users can run this [code](../../tools/get_video_info.py) to obtain the resolution and frame number of the measured video, and can customize the visualization of drawing the polygon area they want and adjust it by themselves.
 The visualization code of the custom polygon region runs as follows:
 ```python
 python get_video_info.py --video_file=demo.mp4 --region_polygon 200 200 400 200 300 400 100 400
 ```

The test result is：

<div align="center">
  <img src="https://user-images.githubusercontent.com/22989727/178769370-03ab1965-cfd1-401b-9902-82620a06e43c.gif" width='600'/>
</div>


## Introduction to the Solution

1. Get the pedestrian detection box of the image/ video input through object detection and multi-object tracking. The detection model is PP-YOLOE, please refer to [PP-YOLOE](../../../../configs/ppyoloe) for details.

2. The multi-object tracking solution is based on [ByteTrack](https://arxiv.org/pdf/2110.06864.pdf) and [OC-SORT](https://arxiv.org/pdf/2203.14360.pdf), and replace the original YOLOX with PP-YOLOE as the detector，and BYTETracker or OC-SORT Tracker as the tracker, please refer to [ByteTrack](../../../../configs/mot/bytetrack) and [OC-SORT](../../../../configs/mot/ocsort).

## Reference
```
@article{zhang2021bytetrack,
  title={ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
  author={Zhang, Yifu and Sun, Peize and Jiang, Yi and Yu, Dongdong and Yuan, Zehuan and Luo, Ping and Liu, Wenyu and Wang, Xinggang},
  journal={arXiv preprint arXiv:2110.06864},
  year={2021}
}
```
