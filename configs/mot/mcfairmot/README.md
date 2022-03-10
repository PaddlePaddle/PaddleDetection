English | [简体中文](README_cn.md)

# MCFairMOT (Multi-class FairMOT)

## Table of Contents
- [Introduction](#Introduction)
- [Model Zoo](#Model_Zoo)
- [Getting Start](#Getting_Start)
- [Citations](#Citations)

## Introduction

MCFairMOT is the Multi-class extended version of [FairMOT](https://arxiv.org/abs/2004.01888).

### PP-Tracking real-time MOT system
In addition, PaddleDetection also provides [PP-Tracking](../../../deploy/pptracking/README.md) real-time multi-object tracking system.
PP-Tracking is the first open source real-time Multi-Object Tracking system, and it is based on PaddlePaddle deep learning framework. It has rich models, wide application and high efficiency deployment.

PP-Tracking supports two paradigms: single camera tracking (MOT) and multi-camera tracking (MTMCT). Aiming at the difficulties and pain points of actual business, PP-Tracking provides various MOT functions and applications such as pedestrian tracking, vehicle tracking, multi-class tracking, small object tracking, traffic statistics and multi-camera tracking. The deployment method supports API and GUI visual interface, and the deployment language supports Python and C++, The deployment platform environment supports Linux, NVIDIA Jetson, etc.

### AI studio public project tutorial
PP-tracking provides an AI studio public project tutorial. Please refer to this [tutorial](https://aistudio.baidu.com/aistudio/projectdetail/3022582).

## Model Zoo
### MCFairMOT Results on VisDrone2019 Val Set
| backbone       | input shape | MOTA | IDF1 |  IDS    |   FPS    | download | config |
| :--------------| :------- | :----: | :----: | :---:  | :------: | :----: |:----: |
| DLA-34         | 1088x608 |  24.3  |  41.6  |  2314  |    -     |[model](https://paddledet.bj.bcebos.com/models/mot/mcfairmot_dla34_30e_1088x608_visdrone.pdparams) | [config](./mcfairmot_dla34_30e_1088x608_visdrone.yml) |
| HRNetV2-W18    | 1088x608 |  20.4  |  39.9  |  2603  |    -     |[model](https://paddledet.bj.bcebos.com/models/mot/mcfairmot_hrnetv2_w18_dlafpn_30e_1088x608_visdrone.pdparams) | [config](./mcfairmot_hrnetv2_w18_dlafpn_30e_1088x608_visdrone.yml) |
| HRNetV2-W18    | 864x480 |  18.2  |  38.7  |  2416  |    -     |[model](https://paddledet.bj.bcebos.com/models/mot/mcfairmot_hrnetv2_w18_dlafpn_30e_864x480_visdrone.pdparams) | [config](./mcfairmot_hrnetv2_w18_dlafpn_30e_864x480_visdrone.yml) |
| HRNetV2-W18    | 576x320 |  12.0  |  33.8  |  2178  |    -     |[model](https://paddledet.bj.bcebos.com/models/mot/mcfairmot_hrnetv2_w18_dlafpn_30e_576x320_visdrone.pdparams) | [config](./mcfairmot_hrnetv2_w18_dlafpn_30e_576x320_visdrone.yml) |

**Notes:**
 - MOTA is the average MOTA of 10 catecories in the VisDrone2019 MOT dataset, and its value is also equal to the average MOTA of all the evaluated video sequences. Here we provide the download [link](https://bj.bcebos.com/v1/paddledet/data/mot/visdrone_mcmot.zip) of the dataset.
 - MCFairMOT used 4 GPUs for training 30 epoches. The batch size is 6 on each GPU for MCFairMOT DLA-34, and 8 for MCFairMOT HRNetV2-W18.

### MCFairMOT Results on VisDrone Vehicle Val Set
| backbone       | input shape | MOTA | IDF1 |  IDS    |   FPS    | download | config |
| :--------------| :------- | :----: | :----: | :---:  | :------: | :----: |:----: |
| DLA-34         | 1088x608 |  37.7  |  56.8  |  199  |    -     |[model](https://paddledet.bj.bcebos.com/models/mot/mcfairmot_dla34_30e_1088x608_visdrone_vehicle_bytetracker.pdparams) | [config](./mcfairmot_dla34_30e_1088x608_visdrone_vehicle_bytetracker.yml) |
| HRNetV2-W18    | 1088x608 |  35.6  |  56.3  |  190  |    -     |[model](https://paddledet.bj.bcebos.com/models/mot/mcfairmot_hrnetv2_w18_dlafpn_30e_1088x608_visdrone_vehicle_bytetracker.pdparams) | [config](./mcfairmot_hrnetv2_w18_dlafpn_30e_1088x608_visdrone_vehicle_bytetracker.yml) |

**Notes:**
 - MOTA is the average MOTA of 4 catecories in the VisDrone Vehicle dataset, and this dataset is extracted from the VisDrone2019 MOT dataset, here we provide the download [link](https://bj.bcebos.com/v1/paddledet/data/mot/visdrone_mcmot_vehicle.zip).
 - The tracker used in MCFairMOT model here is ByteTracker.

### MCFairMOT off-line quantization results on VisDrone Vehicle val-set
|    Model      |  Compression Strategy | Prediction Delay（T4） |Prediction Delay（V100）| Model Configuration File |Compression Algorithm Configuration File |
| :--------------| :------- | :------: | :----: | :----: | :----: |
| DLA-34         | baseline |    41.3  |    21.9 |[Configuration File](./mcfairmot_dla34_30e_1088x608_visdrone_vehicle_bytetracker.yml)|    -     |
| DLA-34         | off-line quantization   |  37.8    |  21.2  |[Configuration File](./mcfairmot_dla34_30e_1088x608_visdrone_vehicle_bytetracker.yml)|[Configuration File](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.3/configs/slim/post_quant/mcfairmot_ptq.yml)|


## Getting Start

### 1. Training
Training MCFairMOT on 4 GPUs with following command
```bash
python -m paddle.distributed.launch --log_dir=./mcfairmot_dla34_30e_1088x608_visdrone/ --gpus 0,1,2,3 tools/train.py -c configs/mot/mcfairmot/mcfairmot_dla34_30e_1088x608_visdrone.yml
```

### 2. Evaluation
Evaluating the track performance of MCFairMOT on val dataset in single GPU with following commands:
```bash
# use weights released in PaddleDetection model zoo
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/mcfairmot/mcfairmot_dla34_30e_1088x608_visdrone.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/mcfairmot_dla34_30e_1088x608_visdrone.pdparams

# use saved checkpoint in training
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/mcfairmot/mcfairmot_dla34_30e_1088x608_visdrone.yml -o weights=output/mcfairmot_dla34_30e_1088x608_visdrone/model_final.pdparams
```
**Notes:**
 - The default evaluation dataset is VisDrone2019 MOT val-set. If you want to change the evaluation dataset, please refer to the following code and modify `configs/datasets/mcmot.yml`：
  ```
  EvalMOTDataset:
    !MOTImageFolder
      dataset_dir: dataset/mot
      data_root: your_dataset/images/val
      keep_ori_im: False # set True if save visualization images or video
  ```
 - Tracking results will be saved in `{output_dir}/mot_results/`, and every sequence has one txt file, each line of the txt file is `frame,id,x1,y1,w,h,score,cls_id,-1,-1`, and you can set `{output_dir}` by `--output_dir`.

### 3. Inference
Inference a vidoe on single GPU with following command:
```bash
# inference on video and save a video
CUDA_VISIBLE_DEVICES=0 python tools/infer_mot.py -c configs/mot/mcfairmot/mcfairmot_dla34_30e_1088x608_visdrone.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/mcfairmot_dla34_30e_1088x608_visdrone.pdparams --video_file={your video name}.mp4  --save_videos
```
**Notes:**
 - Please make sure that [ffmpeg](https://ffmpeg.org/ffmpeg.html) is installed first, on Linux(Ubuntu) platform you can directly install it by the following command:`apt-get update && apt-get install -y ffmpeg`.


### 4. Export model
```bash
CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c configs/mot/mcfairmot/mcfairmot_dla34_30e_1088x608_visdrone.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/mcfairmot_dla34_30e_1088x608_visdrone.pdparams
```

### 5. Using exported model for python inference
```bash
python deploy/pptracking/python/mot_jde_infer.py --model_dir=output_inference/mcfairmot_dla34_30e_1088x608_visdrone --video_file={your video name}.mp4 --device=GPU --save_mot_txts
```
**Notes:**
 - The tracking model is used to predict the video, and does not support the prediction of a single image. The visualization video of the tracking results is saved by default. You can add `--save_mot_txts` to save the txt result file, or `--save_images` to save the visualization images.
 - Each line of the tracking results txt file is `frame,id,x1,y1,w,h,score,cls_id,-1,-1`.

### 6. Off-line quantization

The offline quantization model is calibrated using the VisDrone Vehicle val-set, running as:
```bash
CUDA_VISIBLE_DEVICES=0 python3.7 tools/post_quant.py -c configs/mot/mcfairmot/mcfairmot_dla34_30e_1088x608_visdrone_vehicle_bytetracker.yml --slim_config=configs/slim/post_quant/mcfairmot_ptq.yml
```
**Notes:**
 - Offline quantization uses the VisDrone Vehicle val-set dataset and a 4-class vehicle tracking model by default.

## Citations
```
@article{zhang2020fair,
  title={FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking},
  author={Zhang, Yifu and Wang, Chunyu and Wang, Xinggang and Zeng, Wenjun and Liu, Wenyu},
  journal={arXiv preprint arXiv:2004.01888},
  year={2020}
}

@ARTICLE{9573394,
  author={Zhu, Pengfei and Wen, Longyin and Du, Dawei and Bian, Xiao and Fan, Heng and Hu, Qinghua and Ling, Haibin},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Detection and Tracking Meet Drones Challenge}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TPAMI.2021.3119563}
}

@article{zhang2021bytetrack,
  title={ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
  author={Zhang, Yifu and Sun, Peize and Jiang, Yi and Yu, Dongdong and Yuan, Zehuan and Luo, Ping and Liu, Wenyu and Wang, Xinggang},
  journal={arXiv preprint arXiv:2110.06864},
  year={2021}
}
```
