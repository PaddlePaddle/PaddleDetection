[简体中文](fight_recognition.md) | English

# Fight Recognition Model

## Content
- [1 Quick Start](#quick-start)
- [2 Data Preparation](#data-preparation)
    - [2.1 Dataset Download](#dataset-download)
    - [2.2 Frame Extraction](#frame-extraction)
    - [2.3 Train Set and Validation Set Partition](#trainset-and-validationset-partition)
    - [2.4 Video Segmentation](#video-segmentation)
- [3 Model Training](#model-training)
- [4 Model Evaluation](#model-evaluation)
- [5 Model Export](#model-export)


The real-time humman analysis tool [PP-Human](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/deploy/pphuman) integrates the fight recognition module. This document describes how to complete the training process of the fight recognition model based on [PaddleVideo](https://github.com/PaddlePaddle/PaddleVideo/).

The fight recognition model is based on [PP-TSM](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/pp-tsm.md), and it is modified and adapted based on the training process of the PP-TSM video classification model to complete the model training.

Please refer to the [instruction manual](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/usage.md) to learn how to use the PaddleVideo model library.

| Task | Algorithm | Precision | Inference Speed(ms) | Model Weights | Model Inference and Deployment |
| ---- | ---- | ---------- | ---- | ---- | ---------- |
|  Fight Recognition | PP-TSM | Accuracy：89.06% | T4, 128ms on a 2s' video| [Link](https://videotag.bj.bcebos.com/PaddleVideo-release2.3/ppTSM_fight.pdparams) | [Link](https://videotag.bj.bcebos.com/PaddleVideo-release2.3/ppTSM_fight.zip) |

<a name="quick-start"></a>
## 1 Quick Start

Fight Recognition [Demo](https://videotag.bj.bcebos.com/PaddleVideo-release2.3/fight_demo.mp4).

Firstly, download the inference model and unzip it to `inference` directory. Then execute the following command to recognize whether there is fight action in a given video:

```
cd ${PaddleVideo_root}
python tools/predict.py --input_file fight.avi \
                           --config pptsm_fight_frames_dense.yaml \
                           --model_file inference/ppTSM/ppTSM.pdmodel \
                           --params_file inference/ppTSM/ppTSM.pdiparams \
                           --use_gpu=True \
                           --use_tensorrt=False
```


<a name="data-preparation"></a>
## 2 Data Preparation

PP-TSM is a video classification model based on video clips. The training datas are videos with the suffix `.mp4`、`.avi`  or frame sequence. The format of the label file is `.txt`.

<a name="dataset-download"></a>
### 2.1 Dataset Download

This project is based on the combined data of 6 public datasets related to fighting and violent behavior for model training. The specific information of the public dataset is as follows:

| Dataset | Download Link | Description | Label | Number | Duration |
| ---- | ---- | ---------- | ---- | ---- | ---------- |
|  Surveillance Camera Fight Dataset| https://github.com/sayibet/fight-detection-surv-dataset | clipped videos, monitor perspective | video level | fight：150；non-fight：150 | 2s |
| A Dataset for Automatic Violence Detection in Videos | https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos | clipped videos，indoor recording | video level | violence：115 scenes, 2 seats，230 in total; non-violence：60 scenes，2 seats，120 in total | few seconds |
| Hockey Fight Detection Dataset | https://www.kaggle.com/datasets/yassershrief/hockey-fight-vidoes?resource=download | clipped video，unreal scenes | video level | fight：500；non-fight：500 | 2s |
| Video Fight Detection Dataset | https://www.kaggle.com/datasets/naveenk903/movies-fight-detection-dataset | clipped video，unreal scenes | video level | fight：100；non-fight：101 | 2s |
| Real Life Violence Situations Dataset | https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset | clipped video，unreal scenes | video level | violence：1000；non-violence：1000 | few seconds |
| UBI Abnormal Event Detection Dataset| http://socia-lab.di.ubi.pt/EventDetection/ | unclipped videos, monitor perspective | frame level | fight：216；non-fight：784；clip the video and labeled：fight 1976，non-fight 1630 | original video duration range from few seconds to few minutes, 2s after clipping |

There are 3956 fight (violent) videos and 3501 non-fight (non-violent) videos. The duration of each video is about few seconds.

<a name="frame-extraction"></a>
### 2.2 Frame Extraction

To speed up the training process, extract frames from video as follows:

```bash
cd ${PaddleVideo_root}
python data/ucf101/extract_rawframes.py dataset/ rawframes/ --level 2 --ext mp4
```

Assuming that the fps of video is 30, we can get 30 frames per second in the video.

Videos are stored in the `dataset` directory, fight (violent) videos are stored in `dataset/fight`; non-fight (non-violent) videos are stored in `dataset/nofight`. The `rawframes` directory holds the extracted video frames.

<a name="trainset-and-validationset-partition"></a>
### 2.3 Train Set and Validation Set Partition

The number of validation dataset is 1500，from Surveillance Camera Fight Dataset、A Dataset for Automatic Violence Detection in Videos、UBI Abnormal Event Detection Dataset.

The data can also be divided into training set and validation set according to the following command:

```bash
python split_fight_train_test_dataset.py "rawframes" 2 0.8
```

The file `split_fight_train_test_dataset.py` is in the directory of `deploy/pphuman/tools`.

We can get fight_train_list.txt and fight_val_list.txt finally. The label of fighting is 1 and non-fighting label is 0.

<a name="video-segmentation"></a>
### 2.4 Video Segmentation

For unclipped video, clip it before used for model training. This function `cut_video` in `deploy/pphuman/tools` can clip a given video. The input inclues video path, the start frame and end frame of the clip, and the saved path of the clipped video.

<a name="model-training"></a>
## 3 Model Training
Download the pretrained model：
```bash
wget https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_vd_ssld_v2_pretrained.pdparams
```

Model Training：
```bash
# Single GPU
cd ${PaddleVideo_root}
python main.py --validate -c pptsm_fight_frames_dense.yaml
```

```bash
cd ${PaddleVideo_root}
# multiple GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -B -m paddle.distributed.launch --gpus=“0,1,2,3” \
   --log_dir=log_pptsm_dense  main.py  --validate \
   -c pptsm_fight_frames_dense.yaml
```

<a name="model-evaluation"></a>
## 4 Model Evaluation

```bash
cd ${PaddleVideo_root}
python main.py --test -c pptsm_fight_frames_dense.yaml \
   -w ppTSM_fight_best.pdparams
```

<a name="model-export"></a>
## 5 Model Export

Export the trained model for inference：

```bash
cd ${PaddleVideo_root}
python tools/export_model.py -c pptsm_fight_frames_dense.yaml \
                                -p ppTSM_fight_best.pdparams \
                                -o inference/ppTSM
```
