English | [简体中文](./ppvehicle_violation.md)

# Customized Vehicle Violation

The secondary development of vehicle violation task mainly focuses on the task of lane line segmentation model. PP-LiteSeg model is used to get the lane line data set bdd100k through fine-tune. The process is referred to [PP-LiteSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.7/configs/pp_liteseg/README.md)。

## Data preparation

ppvehicle violation analysis divides the lane line into 4 categories
```
0 Background

1 double yellow line

2 Solid line

3 Dashed line

```

1. For the bdd100k data set, we can combine the processing script provided by [lane_to_mask.py](../../../deploy/pipeline/tools/lane_to_mask.py) and bdd100k [repo](https://github.com/bdd100k/bdd100k) to process the data into the data format required for segmentation.


```
# clone bdd100k：
git clone https://github.com/bdd100k/bdd100k.git

# copy lane_to_mask.py to bdd100k/
cp PaddleDetection/deploy/pipeline/tools/lane_to_mask.py bdd100k/

# preparation bdd100k env
cd bdd100k && pip install -r requirements.txt

#bdd100k to mask
python lane_to_mask.py -i dataset/labels/lane/polygons/lane_train.json -o /output_path

# -i means input path for bdd100k dataset label json，
# -o for output patn

```

2. Organize data and store data in the following format:
```
dataset_root
    |
    |--images  
    |  |--train
    |       |--image1.jpg
    |       |--image2.jpg
    |       |--...
    |  |--val
    |       |--image3.jpg
    |       |--image4.jpg
    |       |--...
    |  |--test
    |       |--image5.jpg
    |       |--image6.jpg
    |       |--...
    |
    |--labels  
    |  |--train
    |       |--label1.jpg
    |       |--label2.jpg
    |       |--...
    |  |--val
    |       |--label3.jpg
    |       |--label4.jpg
    |       |--...
    |  |--test
    |       |--label5.jpg
    |       |--label6.jpg
    |       |--...
    |
```

run [create_dataset_list.py](../../../deploy/pipeline/tools/create_dataset_list.py) create txt file

```
python create_dataset_list.py <dataset_root> #dataset path
                              --type  custom #dataset type，support cityscapes、custom

```

For other data and data annotation, please refer to PaddleSeg [Prepare Custom Datasets](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.7/docs/data/marker/marker_cn.md)


## model training

clone PaddleSeg：
```
git clone https://github.com/PaddlePaddle/PaddleSeg.git
```

prepapation env：
```
cd PaddleSeg
pip install -r requirements.txt
```

### Prepare configuration file
For details, please refer to PaddleSeg [prepare configuration file](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.7/docs/config/pre_config_cn.md).

exp: pp_liteseg_stdc2_bdd100k_1024x512.yml

```
batch_size: 16
iters: 50000

train_dataset:
  type: Dataset
  dataset_root: data/bdd100k    #dataset path  
  train_path: data/bdd100k/train.txt #dataset train txt
  num_classes: 4                     #lane classes
  mode: train
  transforms:
    - type: ResizeStepScaling
      min_scale_factor: 0.5
      max_scale_factor: 2.0
      scale_step_size: 0.25
    - type: RandomPaddingCrop
      crop_size: [512, 1024]
    - type: RandomHorizontalFlip
    - type: RandomAffine
    - type: RandomDistort
      brightness_range: 0.5
      contrast_range: 0.5
      saturation_range: 0.5
    - type: Normalize

val_dataset:
  type: Dataset
  dataset_root: data/bdd100k    #dataset path
  val_path: data/bdd100k/val.txt #dataset val txt
  num_classes: 4
  mode: val
  transforms:
    - type: Normalize

optimizer:
  type: sgd
  momentum: 0.9
  weight_decay: 4.0e-5

lr_scheduler:
  type: PolynomialDecay
  learning_rate: 0.01 #0.01
  end_lr: 0
  power: 0.9

loss:
  types:
    - type: MixedLoss
      losses:
        - type: CrossEntropyLoss
        - type: LovaszSoftmaxLoss
      coef: [0.6, 0.4]
    - type: MixedLoss
      losses:
        - type: CrossEntropyLoss
        - type: LovaszSoftmaxLoss
      coef: [0.6, 0.4]
    - type: MixedLoss
      losses:
        - type: CrossEntropyLoss
        - type: LovaszSoftmaxLoss
      coef: [0.6, 0.4]
  coef: [1, 1,1]


model:
  type: PPLiteSeg
  backbone:
    type: STDC2
    pretrained: https://bj.bcebos.com/paddleseg/dygraph/PP_STDCNet2.tar.gz #Pre-training model
```

### training model

```
#Single GPU training
export CUDA_VISIBLE_DEVICES=0 # Linux
# set CUDA_VISIBLE_DEVICES=0  # Windows
python train.py \
       --config configs/pp_liteseg/pp_liteseg_stdc2_bdd100k_1024x512.yml \
       --do_eval \
       --use_vdl \
       --save_interval 500 \
       --save_dir output

```
### Explanation of training parameters
```
--do_eval    Whether to start the evaluation when saving the model. When starting, the best model will be saved to best according to mIoU model
--use_vdl Whether to enable visualdl to record training data
--save_interval 500  Number of steps between model saving
--save_dir output    Model output path
```

## 2、Multiple GPUs training
if you want to use multiple gpus training, you need to set the environment variable CUDA_VISIBLE_DEVICES is specified as multiple gpus (if not specified, all gpus will be used by default), and the training script will be started using paddle.distributed.launch (because nccl is not supported under windows, multi-card training cannot be used):

```
export CUDA_VISIBLE_DEVICES=0,1,2,3 # 4 gpus
python -m paddle.distributed.launch train.py \
       --config configs/pp_liteseg/pp_liteseg_stdc2_bdd100k_1024x512.yml \
       --do_eval \
       --use_vdl \
       --save_interval 500 \
       --save_dir output
```


After training, you can execute the following commands for performance evaluation:
```
python val.py \
       --config configs/pp_liteseg/pp_liteseg_stdc2_bdd100k_1024x512.yml \
       --model_path output/iter_1000/model.pdparams
```


### Model export

Use the following command to export the trained model as a prediction deployment model.

```
python export.py \
       --config configs/pp_liteseg/pp_liteseg_stdc2_bdd100k_1024x512.yml \
       --model_path output/iter_1000/model.pdparams \
       --save_dir output/inference_model
```


Profile in PP-Vehicle when used `./deploy/pipeline/config/infer_cfg_ppvehicle.yml` set `model_dir` in `LANE_SEG`.
```
LANE_SEG:
  lane_seg_config: deploy/pipeline/config/lane_seg_config.yml  
  model_dir:  output/inference_model
```

Then you can use -->to finish the task of updating the lane line segmentation model.
