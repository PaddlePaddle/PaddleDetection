English | [简体中文](attribute.md)

# Attribute Recognition Modules of PP-Human

Pedestrian attribute recognition has been widely used in the intelligent community, industrial, and transportation monitoring. Many attribute recognition modules have been gathered in PP-Human, including gender, age, hats, eyes, clothing and up to 26 attributes in total. Also, the pre-trained models are offered here and users can download and use them directly.

| Task                 | Algorithm | Precision | Inference Speed(ms) | Download Link                                                                               |
|:---------------------|:---------:|:------:|:------:| :---------------------------------------------------------------------------------: |
| Pedestrian Detection/ Tracking    |  PP-YOLOE | mAP: 56.3 <br> MOTA: 72.0 | Detection: 28ms <br> Tracking：33.1ms | [Download Link](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip) |
| Pedestrian Attribute Analysis   |  StrongBaseline  |  ma: 94.86  | Per Person 2ms | [Download Link](https://bj.bcebos.com/v1/paddledet/models/pipeline/strongbaseline_r50_30e_pa100k.tar) |

1. The precision of detection/ tracking models is obtained by training and testing on the dataset consist of [MOT17](https://motchallenge.net/)，[CrowdHuman](http://www.crowdhuman.org/)，[HIEVE](http://humaninevents.org/) and some business data.
2. The precision of pedestiran attribute analysis is obtained by training and testing on the dataset consist of [PA100k](https://github.com/xh-liu/HydraPlus-Net#pa-100k-dataset)，[RAPv2](http://www.rapdataset.com/rapv2.html)，[PETA](http://mmlab.ie.cuhk.edu.hk/projects/PETA.html) and some business data.
3. The inference speed is T4, the speed of using TensorRT FP16.

## Instruction

1. Download the model from the link in the above table, and unzip it to```./output_inference```.
2. When inputting the image, run the command as follows:
```python
python deploy/pphuman/pipeline.py --config deploy/pphuman/config/infer_cfg.yml \
                                                   --image_file=test_image.jpg \
                                                   --device=gpu \
                                                   --enable_attr=True
```
3. When inputting the video, run the command as follows:
```python
python deploy/pphuman/pipeline.py --config deploy/pphuman/config/infer_cfg.yml \
                                                   --video_file=test_video.mp4 \
                                                   --device=gpu \
                                                   --enable_attr=True
```
4. If you want to change the model path, there are two methods：

    - In ```./deploy/pphuman/config/infer_cfg.yml``` you can configurate different model paths. In attribute recognition models, you can modify the configuration in the field of ATTR.
    - Add `--model_dir` in the command line to change the model path：
```python
python deploy/pphuman/pipeline.py --config deploy/pphuman/config/infer_cfg.yml \
                                                   --video_file=test_video.mp4 \
                                                   --device=gpu \
                                                   --enable_attr=True \
                                                   --model_dir det=ppyoloe/
```

The test result is：

<div width="1000" align="center">
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

4. The model adopted in the attribute recognition is [StrongBaseline](https://arxiv.org/pdf/2107.03576.pdf), where the structure is the multi-class network structure based on ResNet50, and Weighted BCE loss and EMA are introduced for effect optimization.

## Reference
```
@article{jia2020rethinking,
  title={Rethinking of pedestrian attribute recognition: Realistic datasets with efficient method},
  author={Jia, Jian and Huang, Houjing and Yang, Wenjie and Chen, Xiaotang and Huang, Kaiqi},
  journal={arXiv preprint arXiv:2005.11909},
  year={2020}
}
```
