English | [简体中文](ppvehicle_plate.md)

# PP-Vehicle License Plate Recognition Modules

License plate recognition has a very wide range of applications with vehicle identification functions, such as automatic vehicle entrance/exit gates.

PP-Vehicle supports vehicle tracking and license plate recognition. Models are available for download:

| Task                       | Algorithm       | Accuracy     | Inference speed(ms) | Model Download                                                                             |
|:-------------------------- |:---------------:|:------------:|:-------------------:|:------------------------------------------------------------------------------------------:|
| Vehicle Detection/Tracking | PP-YOLOE-l      | mAP: 63.9    | -                   | [Link](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_ppvehicle.zip) |
| Vehicle Detection Model    | ch_PP-OCRv3_det | hmean: 0.979 | -                   | [Link](https://bj.bcebos.com/v1/paddledet/models/pipeline/ch_PP-OCRv3_det_infer.tar.gz)    |
| Vehicle Detection Model    | ch_PP-OCRv3_rec | acc: 0.773   | -                   | [Link](https://bj.bcebos.com/v1/paddledet/models/pipeline/ch_PP-OCRv3_rec_infer.tar.gz)    |

1. The tracking model uses the PPVehicle dataset ( which integrates BDD100K-MOT and UA-DETRAC). The dataset merged car, truck, bus, van from BDD100K-MOT and car, bus, van from UA-DETRAC all into 1 class vehicle(1).
2. License plate detection and recognition models were obtained from fine-tuned PP-OCRv3 model on the CCPD2019 and CCPD2020 mixed license plate datasets.

## How to Use

1. Download models from the above table and unzip it to```PaddleDetection/output_inference```, and modify the model path in the configuration file. Models can also be downloaded automatically by default: Set  enable: True of `VEHICLE_PLATE` in `deploy/pipeline/config/infer_cfg_ppvehicle.yml`

Config Description of `infer_cfg_ppvehicle.yml`：

```
VEHICLE_PLATE:                                                            #模块名称
  det_model_dir: output_inference/ch_PP-OCRv3_det_infer/                  #车牌检测模型路径
  det_limit_side_len: 480                                                 #检测模型单边输入尺寸
  det_limit_type: "max"                                                   #检测模型输入尺寸长短边选择，"max"表示长边
  rec_model_dir: output_inference/ch_PP-OCRv3_rec_infer/                  #车牌识别模型路径
  rec_image_shape: [3, 48, 320]                                           #车牌识别模型输入尺寸
  rec_batch_num: 6                                                        #车牌识别batchsize
  word_dict_path: deploy/pipeline/ppvehicle/rec_word_dict.txt             #OCR模型查询字典
  basemode: "idbased"                                                     #流程类型，'idbased'表示基于跟踪模型
  enable: False                                                           #功能是否开启
```

2. For picture input, the start command is as follows (for more descriptions of the command parameters, please refer to [Quick Start - Parameter Description](. /PPVehicle_QUICK_STARTED.md#41-parameter description)).

   ```python
   #Single image
   python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_ppvehicle.yml \
                                                   --image_file=test_image.jpg \
                                                   --device=gpu \
   #Image folder
   python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_ppvehicle.yml \
    --image_dir=images/ \
    --device=gpu \
   ```

3. For video input, the start command is as follows

```bash
#Single video
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_ppvehicle.yml \
                                                   --video_file=test_video.mp4 \
                                                   --device=gpu \

#Video folder
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_ppvehicle.yml \
                                                   --video_dir=test_videos/ \
                                                   --device=gpu \
```

4. There are two ways to modify the model path

   - Config different model path in ```./deploy/pipeline/config/infer_cfg_ppvehicle.yml```, and modify`VEHICLE_PLATE`field to config license plate recognition model modification
   - **[Recommand]** Add`-o VEHICLE_PLATE.det_model_dir=[YOUR_DETMODEL_PATH] VEHICLE_PLATE.rec_model_dir=[YOUR_RECMODEL_PATH]` to config file in command line.

The test results are as follows:

<div width="600" align="center">
  <img src="../images/ppvehicleplate.jpg"/>
</div>

## Solutions

1. PP-YOLOE is adopted for vehicle detection frame of object detection, multi-object tracking in the picture/video input. For details, please refer to [PP-YOLOE](... /... /... /configs/ppyoloe/README_cn.md)
2. By using the coordinates of the vehicle detection frame, each vehicle's image is intercepted in the input image
3. Use the license plate detection model to identify the location of the license plate in each vehicle screenshot as well as the license plate area. The PP-OCRv3_det model is adopted as the solution, obtained from fine-tuned CCPD dataset in terms of number plate.
4. Use a character recognition model to identify characters in a number plate. The PP-OCRv3_det model is adopted as the solution, obtained from fine-tuned CCPD dataset in terms of number plate.

**Performance optimization measures：**

1. Use a frame skipping strategy to detect license plates every 10 frames to reduce the computing workload.
2. Use the license plate result stabilization strategy to avoid the volatility of single frame results; use all historical license plate recognition results of the same id to gain the most likely result for that id.

## Reference

1. PaddeDetection featured detection model PP-YOLOE](../../../../configs/ppyoloe)。
2. Paddle OCR Model Library [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)。
