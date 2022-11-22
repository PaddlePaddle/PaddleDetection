[简体中文](./detection.md) | English

# Customize Object Detection task

In the practical application of object detection algorithms in a specific industry, additional training is often required for practical use. The project iteration will also need to modify categories. This document details how to use PaddleDetection for a customized object detection algorithm. The process includes data preparation, model optimization roadmap, and modifying the category development process.

## Data Preparation

Customization starts with the preparation of the dataset. We need to collect suitable data for the scenario features, so as to improve the model effect and generalization performance. Then Labeme, LabelImg and other labeling tools will be used to label the object detection bouding boxes and convert the labeling results into COCO or VOC data format. Details please refer to [Data Preparation](../../tutorials/data/PrepareDetDataSet_en.md)

## Model Optimization

### 1. Use customized dataset for training

Modify the corresponding path in the data configuration file based on the prepared data, for example:

configs/dataset/coco_detection.yml`:

```
metric: COCO
num_classes: 80

TrainDataset:
  !COCODataSet
    image_dir: train2017 # Path to the images of the training set relative to the dataset_dir
    anno_path: annotations/instances_train2017.json # Path to the annotation file of the training set relative to the dataset_dir
    dataset_dir: dataset/coco # Path to the dataset relative to the PaddleDetection path
    data_fields: ['image', 'gt_bbox', 'gt_class', 'is_crowd']

EvalDataset:
  !COCODataSet
    image_dir: val2017 # Path to the images of the evaldataset set relative to the dataset_dir
    anno_path: annotations/instances_val2017.json # Path to the annotation file of the evaldataset relative to the dataset_dir
    dataset_dir: dataset/coco # Path to the dataset relative to the PaddleDetection path

TestDataset:
  !ImageFolder
    anno_path: annotations/instances_val2017.json # also support txt (like VOC's label_list.txt) # Path to the annotation files relative to dataset_di.
    dataset_dir: dataset/coco # if set, anno_path will be 'dataset_dir/anno_path' # Path to the dataset relative to the PaddleDetection path
```

Once the configuration changes are completed, the training evaluation can be started with the following command

```
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/yolov3/yolov3_mobilenet_v1_270e_coco.yml --eval
```

More details please refer to [Getting Started for PaddleDetection](../../tutorials/GETTING_STARTED_cn.md)

###

### 2. Load the COCO model as pre-training

The currently provided pre-trained models in PaddleDetection's configurations are weights from the ImageNet dataset, loaded into the backbone network of the detection algorithm. For practical use, it is recommended to load the weights trained on the COCO dataset, which can usually provide a large improvement to the model accuracy. The method is as follows.

#### 1) Set pre-training weight path

The trained model weights for the COCO dataset are saved in the configuration folder of each algorithm, for example, PP-YOLOE-l COCO dataset weights are provided under `configs/ppyoloe`: [Link](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams) The configuration file sets`pretrain_weights: https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams`

#### 2) Modify hyperparameters

After loading the COCO pre-training weights, the learning rate hyperparameters need to be modified, for example

In `configs/ppyoloe/_base_/optimizer_300e.yml`:

```
epoch: 120 # The original configuration is 300 epoch, after loading COCO weights, the iteration number can be reduced appropriately

LearningRate:
 base_lr: 0.005 # The original configuration is 0.025, after loading COCO weights, the learning rate should be reduced.
 schedulers:
 - !CosineDecay
 max_epochs: 144 # Modify based on the number of epochs
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
