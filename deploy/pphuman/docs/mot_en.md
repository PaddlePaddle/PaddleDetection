# Detection and Tracking Module of PP-Human

Pedestrian detection and tracking is widely used in the intelligent community, industrial inspection, transportation monitoring and so on. PP-Human has the detection and tracking module, which is fundamental to keypoint detection, attribute action recognition, etc. Users enjoy easy access to pretrained models here.

| Task                 | Algorithm | Precision | Inference Speed(ms) | Download Link                                                                               |
|:---------------------|:---------:|:------:|:------:| :---------------------------------------------------------------------------------: |
| Pedestrian Detection/ Tracking    |  PP-YOLOE | mAP: 56.3 <br> MOTA: 72.0 | Detection: 28ms <br> Tracking：33.1ms | [Download Link](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip) |

1. The precision of the pedestrian detection/ tracking model is obtained by trainning and testing on [MOT17](https://motchallenge.net/), [CrowdHuman](http://www.crowdhuman.org/), [HIEVE](http://humaninevents.org/) and some business data.
2. The inference speed is the speed of using TensorRT FP16 on T4, the total number of data pre-training, model inference, and post-processing.

## How to Use

1. Download models from the links of the above table and unizp them to ```./output_inference```.
2. During the image input, the start command is as follows:
```python
python deploy/pphuman/pipeline.py --config deploy/pphuman/config/infer_cfg.yml \
                                                   --image_file=test_image.jpg \
                                                   --device=gpu
```
3. In the video input, the start command is as follows:
```python
python deploy/pphuman/pipeline.py --config deploy/pphuman/config/infer_cfg.yml \
                                                   --video_file=test_video.mp4 \
                                                   --device=gpu
```
4. There are two ways to modify the model path:

     - In `./deploy/pphuman/config/infer_cfg.yml`, you can configurate different model paths，which is proper only if you match keypoint models and action recognition models with the fields of `DET` and `MOT` respectively, and modify the corresponding path of each field into the expected path.
    - Add `--model_dir` in the command line to revise the model path:

```python
python deploy/pphuman/pipeline.py --config deploy/pphuman/config/infer_cfg.yml \
                                                   --video_file=test_video.mp4 \
                                                   --device=gpu \
                                                   --model_dir det=ppyoloe/
                                                   --do_entrance_counting \
                                                   --draw_center_traj

```
**Note:**

 - `--do_entrance_counting` is whether to calculate flow at the gateway, and the default setting is False.
 - `--draw_center_traj` means whether to draw the track, and the default setting is False. It's worth noting that the test video of track drawing should be filmed by the still camera.
The test result is：

<div width="1000" align="center">
  <img src="./images/mot.gif"/>
</div>

Data source and copyright owner：Skyinfor Technology. Thanks for the provision of actual scenario data, which are only used for academic research here.


## Introduction to the Solution

1. Get the pedestrian detection box of the image/ video input through object detection and multi-object tracking. The adopted model is PP-YOLOE, and for details, please refer to [PP-YOLOE](../../../configs/ppyoloe).

2. The multi-object tracking model solution is based on [ByteTrack](https://arxiv.org/pdf/2110.06864.pdf), and replace the original YOLOX with P-YOLOE as the detector，and BYTETracker as the tracker.

## Reference
```
@article{zhang2021bytetrack,
  title={ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
  author={Zhang, Yifu and Sun, Peize and Jiang, Yi and Yu, Dongdong and Yuan, Zehuan and Luo, Ping and Liu, Wenyu and Wang, Xinggang},
  journal={arXiv preprint arXiv:2110.06864},
  year={2021}
}
```
