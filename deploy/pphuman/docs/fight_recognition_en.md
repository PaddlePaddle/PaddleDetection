[简体中文](fight_recognition.md) | English

# Fight Recognition Module of PP-Human
With the wider application of surveillance cameras, it is time-consuming and labor-intensive and inefficient to manually check whether there are abnormal behaviors such as fighting. AI + security assistant smart security. A fight recognition module is integrated into PP-Human to identify whether there is fighting in the video. We provide pre-trained model that users can download and use directly.

| Task | Algorithm | Precision | Inference Speed(ms) | Model Weights | Model Inference and Deployment |
| ---- | ---- | ---------- | ---- | ---- | ---------- |
|  Fight Recognition | PP-TSM | Accuracy：89.06% | T4, 128ms on a 2s' video| [Link](https://videotag.bj.bcebos.com/PaddleVideo-release2.3/ppTSM_fight.pdparams) | [Link](https://videotag.bj.bcebos.com/PaddleVideo-release2.3/ppTSM_fight.zip) |

The fight recognition model is trained based on 6 public datasets：Surveillance Camera Fight Dataset、A Dataset for Automatic Violence Detection in Videos、Hockey Fight Detection Dataset、Video Fight Detection Dataset、Real Life Violence Situations Dataset、UBI Abnormal Event Detection Dataset.

## Instruction
1. Download the inference model from the link in the above table, and unzip it to `./output_inference`;
2. Modify the files name to `model.pdiparams、model.pdiparams.info and model.pdmodel` in the directory of `ppTSM`;
3. Modify the config file `deploy/pphuman/config/infer_cfg_pphuman.yml` and change `enable` from `False` to `True` in `VIDEO_ACTION`;
4. Input a video and run the command as follows:
```
python deploy/pphuman/pipeline.py --config deploy/pphuman/config/infer_cfg.yml \
                                                   --video_file=test_video.mp4 \
                                                   --device=gpu
```

The test result is：

<div width="1000" align="center">
  <img src="./images/fight_demo.gif"/>
</div>

Data Source and Copyright：Surveillance Camera Fight Dataset.

## Introduction to the Solution

The fight recognition model is based on [PP-TSM](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/pp-tsm.md), and it is modified and adapted based on the training process of the PP-TSM video classification model to complete the model training.
