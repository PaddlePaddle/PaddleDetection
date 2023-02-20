[简体中文](./pphuman_mot.md) | English

# Customized multi-object tracking task

When applying multi-object tracking algorithms in industrial applications, there will be inevitable demands for customized types of multi-object tracking or optimization of existing multi-object tracking models to improve the effectiveness of the models in specific scenarios. In this document, we present examples of how to choose a multi-object tracking solution based on the expected identified behavior, and how to use PaddleDetection for further development of multi-object tracking algorithms, including data preparation, model optimization ideas, and the development process of tracking category modification.

## Data Preparation

The multi-object tracking model scheme uses [ByteTrack](https://arxiv.org/pdf/2110.06864.pdf), which adopts PP-YOLOE to replace the original YOLOX as a detector and BYTETracker as a tracker, for details, please refer to [ByteTrack](... /... /... /configs/mot/bytetrack). The original ByteTrack only supports single pedestrian category, while PaddleDetection supports multiple categories for simultaneous tracking. Training ByteTrack, which is the process of training the detector, only requires the detection annotations to be prepared, and does not require ReID annotation information, i.e., it can be done as pure detection. The dataset should preferably be extracted from continuous video rather than a collection of unrelated images.

Customization starts with the preparation of the dataset. We need to collect suitable data for the scenario features, so as to improve the model effect and generalization performance. Then Labeme, LabelImg and other labeling tools will be used to label the object detection frame and convert the labeling results into COCO or VOC data format. Details please refer to [Data Preparation](../../tutorials/data/README.md)

## Model Optimization

### 1. Use customized data set for training

The dataset used by the ByteTrack tracking solution only needs detection annotations. Refer to [MOT dataset preparation](... /... /... /configs/mot) and [MOT dataset tutorial](docs/tutorials/data/PrepareMOTDataSet.md).

```
# Single card training
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/ppyoloe/ppyoloe_crn_l_300e_coco.yml --eval --amp

# Multi-card training
python -m paddle.distributed.launch --log_dir=log_dir --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/ppyoloe/ppyoloe_crn_l_300e_coco.yml --eval --amp
```

More details please refer to [Getting Started for PaddleDetection](../../tutorials/GETTING_STARTED_cn.md) and [ByteTrack](../../../configs/mot/bytetrack/detector)

### 2. Load the COCO model as the pre-trained model

The currently provided pre-trained models in PaddleDetection's configurations are weights from the ImageNet dataset, loaded into the backbone network of the detection algorithm. For practical use, it is recommended to load the weights trained on the COCO dataset, which can usually provide a large improvement to the model accuracy. The method is as follows.

#### 1) Set pre-training weight path

The trained model weights for the COCO dataset are saved in the configuration folder of each algorithm, for example, PP-YOLOE-l COCO dataset weights are provided under `configs/ppyoloe`: [Link](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams) The configuration file sets`pretrain_weights: https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams`

#### 2) Modify hyperparameters

After loading the COCO pre-training weights, the learning rate hyperparameters need to be modified, for example

In `configs/ppyoloe/*base*/optimizer_300e.yml`:

```
epoch: 120 # The original configuration is 300 epoch, after loading COCO weights, the iteration number can be reduced appropriately

LearningRate:
base_lr: 0.005 # The original configuration is 0.025, after loading COCO weights, the learning rate should be reduced.
  schedulers:
    - !CosineDecay
      max_epochs: 144 # Modified according to the number of epochs, usually 1.2 times the number of epochs
    - LinearWarmup
      start_factor: 0.
      epochs: 5
```

## Modify categories

When the actual application scenario category changes, the data configuration file needs to be modified, for example in `configs/datasets/coco_detection.yml`:

```
metric: COCO
num_classes: 10 # original class 80
```

After the configuration changes are completed, the COCO pre-training weights can also be loaded. PaddleDetection supports automatic loading of shape-matching weights, and weights that do not match the shape are automatically ignored, so no other modifications are needed.
