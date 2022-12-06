English | [简体中文](pphuman_attribute.md)

# Attribute Recognition Modules of PP-Human

Pedestrian attribute recognition has been widely used in the intelligent community, industrial, and transportation monitoring. Many attribute recognition modules have been gathered in PP-Human, including gender, age, hats, eyes, clothing and up to 26 attributes in total. Also, the pre-trained models are offered here and users can download and use them directly.

| Task                 | Algorithm | Precision | Inference Speed(ms) | Download Link                                                                               |
|:---------------------|:---------:|:------:|:------:| :---------------------------------------------------------------------------------: |
| High-Precision Model    |  PP-HGNet_small  |  mA: 95.4  | per person 1.54ms | [Download](https://bj.bcebos.com/v1/paddledet/models/pipeline/PPHGNet_small_person_attribute_954_infer.tar) |
| Fast Model    |  PP-LCNet_x1_0  |  mA: 94.5  | per person 0.54ms | [Download](https://bj.bcebos.com/v1/paddledet/models/pipeline/PPLCNet_x1_0_person_attribute_945_infer.tar) |
| Balanced Model    |  PP-HGNet_tiny  |  mA: 95.2  | per person 1.14ms | [Download](https://bj.bcebos.com/v1/paddledet/models/pipeline/PPHGNet_tiny_person_attribute_952_infer.tar) |

1. The precision of pedestiran attribute analysis is obtained by training and testing on the dataset consist of [PA100k](https://github.com/xh-liu/HydraPlus-Net#pa-100k-dataset)，[RAPv2](http://www.rapdataset.com/rapv2.html)，[PETA](http://mmlab.ie.cuhk.edu.hk/projects/PETA.html) and some business data.
2. The inference speed is V100, the speed of using TensorRT FP16.
3. This model of Attribute is based on the result of tracking, please download tracking model in the [Page of Mot](./pphuman_mot_en.md). The High precision and Faster model are both available.
4. You should place the model unziped in the directory of `PaddleDetection/output_inference/`.

## Instruction

1. Download the model from the link in the above table, and unzip it to```./output_inference```, and set the "enable: True" in ATTR of infer_cfg_pphuman.yml

The meaning of configs of `infer_cfg_pphuman.yml`：
```
ATTR:                                                                     #module name
  model_dir: output_inference/PPLCNet_x1_0_person_attribute_945_infer/    #model path
  batch_size: 8                                                           #maxmum batchsize when inference
  enable: False                                                           #whether to enable this model
```

2. When inputting the image, run the command as follows (please refer to [QUICK_STARTED-Parameters](./PPHuman_QUICK_STARTED.md#41-参数说明) for more details):
```python
#single image
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_pphuman.yml \
                                                   --image_file=test_image.jpg \
                                                   --device=gpu \

#image directory
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_pphuman.yml \
                                                   --image_dir=images/ \
                                                   --device=gpu \

```
3. When inputting the video, run the command as follows:
```python
#a single video file
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_pphuman.yml \
                                                   --video_file=test_video.mp4 \
                                                   --device=gpu \

#directory of videos
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_pphuman.yml \
                                                   --video_dir=test_videos/ \
                                                   --device=gpu \
```
4. If you want to change the model path, there are two methods：

    - The first: In ```./deploy/pipeline/config/infer_cfg_pphuman.yml``` you can configurate different model paths. In attribute recognition models, you can modify the configuration in the field of ATTR.
    - The second: Add `-o ATTR.model_dir` in the command line following the --config to change the model path：
```python
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_pphuman.yml \
                                                   -o ATTR.model_dir=output_inference/PPLCNet_x1_0_person_attribute_945_infer/\
                                                   --video_file=test_video.mp4 \
                                                   --device=gpu
```

The test result is：

<div width="600" align="center">
  <img src="https://user-images.githubusercontent.com/48054808/159898428-5bda0831-7249-4889-babd-9165f26f664d.gif"/>
</div>

Data Source and Copyright：Skyinfor Technology. Thanks for the provision of actual scenario data, which are only used for academic research here.

## Introduction to the Solution

1. The PP-YOLOE model is used to handle detection boxs of input images/videos from object detection/ multi-object tracking. For details, please refer to the document [PP-YOLOE](../../../configs/ppyoloe).
2. Capture every pedestrian in the input images with the help of coordiantes of detection boxes.
3. Analyze the listed labels of pedestirans through attribute recognition. They are the same as those in the PA100k dataset. The label list is as follows:
```
- Gender
- Age: Less than 18; 18-60; Over 60
- Orientation: Front; Back; Side
- Accessories: Glasses; Hat; None
- HoldObjectsInFront: Yes; No
- Bag: BackPack; ShoulderBag; HandBag
- TopStyle: UpperStride; UpperLogo; UpperPlaid; UpperSplice
- BottomStyle: LowerStripe; LowerPattern
- ShortSleeve: Yes; No
- LongSleeve: Yes; No
- LongCoat: Yes; No
- Trousers: Yes; No
- Shorts: Yes; No
- Skirt&Dress: Yes; No
- Boots: Yes; No
```

4. The model adopted in the attribute recognition is [StrongBaseline](https://arxiv.org/pdf/2107.03576.pdf), where the structure is the multi-class network structure based on PP-HGNet、PP-LCNet, and Weighted BCE loss is introduced for effect optimization.

## Reference
```
@article{jia2020rethinking,
  title={Rethinking of pedestrian attribute recognition: Realistic datasets with efficient method},
  author={Jia, Jian and Huang, Houjing and Yang, Wenjie and Chen, Xiaotang and Huang, Kaiqi},
  journal={arXiv preprint arXiv:2005.11909},
  year={2020}
}
```
