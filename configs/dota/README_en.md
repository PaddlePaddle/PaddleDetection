# S2ANet Model

## Content
- [S2ANet Model](#s2anet-model)
  - [Content](#content)
  - [Introduction](#introduction)
  - [Prepare Data](#prepare-data)
    - [DOTA data](#dota-data)
    - [Customize Data](#customize-data)
  - [Start Training](#start-training)
    - [1. Install the rotating frame IOU and calculate the OP](#1-install-the-rotating-frame-iou-and-calculate-the-op)
    - [2. Train](#2-train)
    - [3. Evaluation](#3-evaluation)
    - [4. Prediction](#4-prediction)
    - [5. DOTA Data evaluation](#5-dota-data-evaluation)
  - [Model Library](#model-library)
    - [S2ANet Model](#s2anet-model-1)
  - [Predict Deployment](#predict-deployment)
  - [Citations](#citations)

## Introduction

[S2ANet](https://arxiv.org/pdf/2008.09397.pdf) is used to detect rotating frame's model, required use of PaddlePaddle 2.1.1(can be installed using PIP) or proper [develop version](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html#whl-release).


## Prepare Data

### DOTA data
[DOTA Dataset] is a dataset of object detection in aerial images, which contains 2806 images with a resolution of 4000x4000 per image.

|  Data version  |  categories  |   images   |  size  |    instances    |     annotation method     |
|:--------:|:-------:|:---------:|:---------:| :---------:| :------------: |
|   v1.0   |   15    |   2806    | 800~4000  |   118282    |   OBB + HBB     |
|   v1.5   |   16    |   2806    | 800~4000  |   400000    |   OBB + HBB     |

Note: OBB annotation is an arbitrary quadrilateral; The vertices are arranged in clockwise order. The HBB annotation mode is the outer rectangle of the indicator note example.

There were 2,806 images in the DOTA dataset, including 1,411 images as a training set, 458 images as an evaluation set, and the remaining 937 images as a test set.

If you need to cut the image data, please refer to the [DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit).

After setting `crop_size=1024, stride=824, gap=200` parameters to cut data, there are 15,749 images in the training set, 5,297 images in the evaluation set, and 10,833 images in the test set.

### Customize Data

There are two ways to annotate data:

- The first is a tagging rotating rectangular, can pass rotating rectangular annotation tool [roLabelImg](https://github.com/cgvict/roLabelImg) to describe rotating rectangular box.

- The second is to mark the quadrilateral, through the script into an external rotating rectangle, so that the obtained mark may have a certain error with the real object frame.

Then convert the annotation result into coco annotation format, where each `bbox` is in the format of `[x_center, y_center, width, height, angle]`, where the angle is expressed in radians.

Reference [spinal disk dataset](https://aistudio.baidu.com/aistudio/datasetdetail/85885), we divide dataset into training set (230), the test set (57), data address is: [spine_coco](https://paddledet.bj.bcebos.com/data/spine_coco.tar). The dataset has a small number of images, which can be used to train the S2ANet model quickly.


## Start Training

### 1. Install the rotating frame IOU and calculate the OP

Rotate box IoU calculate [ext_op](../../ppdet/ext_op) is a reference PaddlePaddle [custom external operator](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/07_new_op/new_custom_op.html).

To use the rotating frame IOU to calculate the OP, the following conditions must be met:
- PaddlePaddle >= 2.1.1
- GCC == 8.2

Docker images are recommended[paddle:2.1.1-gpu-cuda10.1-cudnn7](registry.baidubce.com/paddlepaddle/paddle:2.1.1-gpu-cuda10.1-cudnn7)。

Run the following command to download the image and start the container:
```
sudo nvidia-docker run -it --name paddle_s2anet -v $PWD:/paddle --network=host registry.baidubce.com/paddlepaddle/paddle:2.1.1-gpu-cuda10.1-cudnn7 /bin/bash
```

If the PaddlePaddle are installed in the mirror, go to python3.7 and run the following code to check whether the PaddlePaddle are installed properly:
```
import paddle
print(paddle.__version__)
paddle.utils.run_check()
```

enter `ppdet/ext_op` directory, install：
```
python3.7 setup.py install
```

In Windows, perform the following steps to install it：

（1）Visual Studio (version required >= Visual Studio 2015 Update3);

（2）Go to Start --> Visual Studio 2017 --> X64 native Tools command prompt for VS 2017;

（3）Setting Environment Variables：`set DISTUTILS_USE_SDK=1`

（4）Enter `PaddleDetection/ppdet/ext_op` directory，use `python3.7 setup.py install` to install。

After the installation, test whether the custom OP can compile normally and calculate the results:
```
cd PaddleDetecetion/ppdet/ext_op
python3.7 test.py
```

### 2. Train
**Attention:**
In the configuration file, the learning rate is set based on the eight-card GPU training. If the single-card GPU training is used, set the learning rate to 1/8 of the original value.

Single GPU Training
```bash
export CUDA_VISIBLE_DEVICES=0
python3.7 tools/train.py -c configs/dota/s2anet_1x_spine.yml
```

Multiple GPUs Training
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3.7 -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/dota/s2anet_1x_spine.yml
```

You can use `--eval`to enable train-by-test.

### 3. Evaluation
```bash
python3.7 tools/eval.py -c configs/dota/s2anet_1x_spine.yml -o weights=output/s2anet_1x_spine/model_final.pdparams

# Use a trained model to evaluate
python3.7 tools/eval.py -c configs/dota/s2anet_1x_spine.yml -o weights=https://paddledet.bj.bcebos.com/models/s2anet_1x_spine.pdparams
```
**Attention:**
(1) The DOTA dataset is trained together with train and val data as a training set, and the evaluation dataset configuration needs to be customized when evaluating the DOTA dataset.

(2) Bone dataset is transformed from segmented data. As there is little difference between different types of discs for detection tasks, and the score obtained by S2ANET algorithm is low, the default threshold for evaluation is 0.5, a low mAP is normal. You are advised to view the detection result visually.

### 4. Prediction
Executing the following command will save the image prediction results to the `output` folder.
```bash
python3.7 tools/infer.py -c configs/dota/s2anet_1x_spine.yml -o weights=output/s2anet_1x_spine/model_final.pdparams --infer_img=demo/39006.jpg --draw_threshold=0.3
```
Prediction using models that provide training:
```bash
python3.7 tools/infer.py -c configs/dota/s2anet_1x_spine.yml -o weights=https://paddledet.bj.bcebos.com/models/s2anet_1x_spine.pdparams --infer_img=demo/39006.jpg --draw_threshold=0.3
```

### 5. DOTA Data evaluation
Execute the following command, will save each image prediction result in `output` folder txt text with the same folder name.
```
python3.7 tools/infer.py -c configs/dota/s2anet_alignconv_2x_dota.yml -o weights=./weights/s2anet_alignconv_2x_dota.pdparams  --infer_dir=dota_test_images --draw_threshold=0.05 --save_txt=True --output_dir=output
```
Please refer to [DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit) generate assessment files, Assessment file format, please refer to [DOTA Test](http://captain.whu.edu.cn/DOTAweb/tasks.html), and generate the zip file, each class a txt file, every row in the txt file format for: `image_id score x1 y1 x2 y2 x3 y3 x4 y4` You can also reference the `dataset/dota_coco/dota_generate_test_result.py` script to generate an evaluation file and submit it to the server.

## Model Library

### S2ANet Model

|     Model     |  Conv Type  |   mAP    |   Model Download   |   Configuration File   |
|:-----------:|:----------:|:--------:| :----------:| :---------: |
|   S2ANet    |   Conv     |   71.42  |  [model](https://paddledet.bj.bcebos.com/models/s2anet_conv_2x_dota.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/dota/s2anet_conv_2x_dota.yml)                   |
|   S2ANet    |  AlignConv |   74.0   |  [model](https://paddledet.bj.bcebos.com/models/s2anet_alignconv_2x_dota.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/dota/s2anet_alignconv_2x_dota.yml)                   |

**Attention:** `multiclass_nms` is used here, which is slightly different from the original author's use of NMS.


## Predict Deployment

The inputs of the `multiclass_nms` operator in Paddle support quadrilateral inputs, so deployment can be done without relying on the rotating frame IOU operator.

Please refer to the deployment tutorial[Predict deployment](../../deploy/README_en.md)


## Citations
```
@article{han2021align,  
  author={J. {Han} and J. {Ding} and J. {Li} and G. -S. {Xia}},  
  journal={IEEE Transactions on Geoscience and Remote Sensing},  
  title={Align Deep Features for Oriented Object Detection},  
  year={2021},
  pages={1-11},  
  doi={10.1109/TGRS.2021.3062048}}

@inproceedings{xia2018dota,
  title={DOTA: A large-scale dataset for object detection in aerial images},
  author={Xia, Gui-Song and Bai, Xiang and Ding, Jian and Zhu, Zhen and Belongie, Serge and Luo, Jiebo and Datcu, Mihai and Pelillo, Marcello and Zhang, Liangpei},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3974--3983},
  year={2018}
}
```
