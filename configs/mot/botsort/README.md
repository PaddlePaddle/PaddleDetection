English | [简体中文](README_cn.md)

# BOT_SORT (BoT-SORT: Robust Associations Multi-Pedestrian Tracking)

## content
- [introduction](#introduction)
- [model zoo](#modelzoo)
- [Quick Start](#QuickStart)
- [Citation](Citation)

## introduction
[BOT_SORT](https://arxiv.org/pdf/2206.14651v2.pdf)(BoT-SORT: Robust Associations Multi-Pedestrian Tracking). The configuration of common detectors is provided here for reference. Because different training data sets, input scales, number of training epochs, NMS threshold settings, etc. will lead to differences in model accuracy and performance, please adapt according to your needs

## modelzoo

### BOT_SORT在MOT-17 half Val Set

|  Dataset      |  detector     | input size  | detector mAP  |  MOTA  |  IDF1  |  config |
| :--------         | :-----      | :----:  | :------:  | :----: |:-----: |:----:   |
| MOT-17 half train | PP-YOLOE-l  | 640x640 |  52.7    |  55.5  |  64.2 |[config](./botsort_ppyoloe.yml) |


**Attention:**
  - Model weight download link in the configuration file ` ` ` det_ Weights ` ` `, run the verification command to automatically download.
  - **MOT17-half train** is a data set composed of pictures and labels of the first half frames of each video in the MOT17 train sequence (7 in total). To verify the accuracy, we can use the **MOT17-half val** to eval，It is composed of the second half frame of each video，download [link](https://bj.bcebos.com/v1/paddledet/data/mot/MOT17.zip)，decompression `dataset/mot/`

  - BOT_ SORT training is a separate detector training MOT dataset, reasoning is to assemble a tracker to evaluate MOT indicators, and a separate detection model can also evaluate detection indicators.
  - BOT_SORT export deployment is to export the detection model separately and then assemble the tracker for operation. Refer to [PP-Tracking](../../../deploy/pptracking/python)。
  - BOT_SORT is the main scheme for PP Human, PP Vehicle and other pipelines to analyze the project tracking direction. For specific use, please refer to [Pipeline](../../../deploy/pipeline) and [MOT](../../../deploy/pipeline/docs/tutorials/pphuman_mot.md).


## QuickStart

### 1. train
Start training and evaluation with the following command
```bash
#Single gpu
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/mot/bytetrack/detector/ppyoloe_crn_l_36e_640x640_mot17half.yml --eval --amp

#Multi gpu
python -m paddle.distributed.launch --log_dir=ppyoloe --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/mot/bytetrack/detector/ppyoloe_crn_l_36e_640x640_mot17half.yml --eval --amp
```

### 2. evaluate
#### 2.1 detection
```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/mot/bytetrack/detector/ppyoloe_crn_l_36e_640x640_mot17half.yml
```

**Attention:**
 - eval detection use ```tools/eval.py```，eval mot use ```tools/eval_mot.py```.

#### 2.2 mot
```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/botsort/botsort_ppyoloe.yml --scaled=True
```
**Attention:**
 - `--scaled` indicates whether the coordinates of the output results of the model have been scaled back to the original drawing. If the detection model used is JDE YOLOv3, it is false. If the universal detection model is used, it is true. The default value is false.
 - mot result save `{output_dir}/mot_results/`,each video sequence in it corresponds to a txt, and each line of information in each txt file is `frame,id,x1,y1,w,h,score,-1,-1,-1`, and `{output_dir}` could  use `--output_dir` to set.

### 3. export detection model

```bash
python tools/export_model.py -c configs/mot/bytetrack/detector/ppyoloe_crn_l_36e_640x640_mot17half.yml --output_dir=output_inference -o weights=https://bj.bcebos.com/v1/paddledet/models/mot/ppyoloe_crn_l_36e_640x640_mot17half.pdparams
```

### 4. Use the export model to predict

```bash
# download demo video
wget https://bj.bcebos.com/v1/paddledet/data/mot/demo/mot17_demo.mp4

CUDA_VISIBLE_DEVICES=0 python deploy/pptracking/python/mot_sde_infer.py --model_dir=output_inference/ppyoloe_crn_l_36e_640x640_mot17half --tracker_config=deploy/pptracking/python/tracker_config.yml --video_file=mot17_demo.mp4 --device=GPU --threshold=0.5
```
**Attention:**
 - You must fix `tracker_config.yml` tracker `type: BOTSORTTracker`，if you want to use BOT_SORT.
 - The tracking model is used to predict videos. It does not support prediction of a single image. By default, the videos with visualized tracking results are saved. You can add `--save_mot_txts` (save a txt for each video) or `--save_mot_txt_per_img`(Save a txt for each image)  or `--save_images` save the visualization picture of tracking results.
 - Each line of the trace result txt file format `frame,id,x1,y1,w,h,score,-1,-1,-1`。


## Citation
```
@article{aharon2022bot,
  title={BoT-SORT: Robust Associations Multi-Pedestrian Tracking},
  author={Aharon, Nir and Orfaig, Roy and Bobrovsky, Ben-Zion},
  journal={arXiv preprint arXiv:2206.14651},
  year={2022}
}
```
