简体中文 | [English](README_en.md)

# KeyPoint模型系列



## 简介

-    PaddleDetection KeyPoint部分紧跟业内最新最优算法方案，包含Top-Down、BottomUp两套方案，以满足用户的不同需求。

<div align="center">
  <img src="./football_keypoint.gif" width='800'/>
</div>



####   Model Zoo
COCO数据集
| 模型              | 输入尺寸 | AP(coco val) |                           模型下载                           | 配置文件                                                    |
| :---------------- | -------- | :----------: | :----------------------------------------------------------: | ----------------------------------------------------------- |
| HigherHRNet-w32       | 512      |     67.1     | [higherhrnet_hrnet_w32_512.pdparams](https://paddledet.bj.bcebos.com/models/keypoint/higherhrnet_hrnet_w32_512.pdparams) | [config](./higherhrnet/higherhrnet_hrnet_w32_512.yml)       |
| HigherHRNet-w32       | 640      |     68.3     | [higherhrnet_hrnet_w32_640.pdparams](https://paddledet.bj.bcebos.com/models/keypoint/higherhrnet_hrnet_w32_640.pdparams) | [config](./higherhrnet/higherhrnet_hrnet_w32_640.yml)       |
| HigherHRNet-w32+SWAHR | 512      |     68.9     | [higherhrnet_hrnet_w32_512_swahr.pdparams](https://paddledet.bj.bcebos.com/models/keypoint/higherhrnet_hrnet_w32_512_swahr.pdparams) | [config](./higherhrnet/higherhrnet_hrnet_w32_512_swahr.yml) |
| HRNet-w32             | 256x192  |     76.9     | [hrnet_w32_256x192.pdparams](https://paddledet.bj.bcebos.com/models/keypoint/hrnet_w32_256x192.pdparams) | [config](./hrnet/hrnet_w32_256x192.yml)                     |
| HRNet-w32             | 384x288  |     77.8     | [hrnet_w32_384x288.pdparams](https://paddledet.bj.bcebos.com/models/keypoint/hrnet_w32_384x288.pdparams) | [config](./hrnet/hrnet_w32_384x288.yml)                     |
| HRNet-w32+DarkPose             | 256x192  |     78.0     | [dark_hrnet_w32_256x192.pdparams](https://paddledet.bj.bcebos.com/models/keypoint/dark_hrnet_w32_256x192.pdparams) | [config](./hrnet/dark_hrnet_w32_256x192.yml)                     |
| HRNet-w32+DarkPose             | 384x288  |     78.3     | [dark_hrnet_w32_384x288.pdparams](https://paddledet.bj.bcebos.com/models/keypoint/dark_hrnet_w32_384x288.pdparams) | [config](./hrnet/dark_hrnet_w32_384x288.yml)                     |
| WiderNaiveHRNet-18         | 256x192  |     67.6(+DARK 68.4)     | [wider_naive_hrnet_18_256x192_coco.pdparams](https://bj.bcebos.com/v1/paddledet/models/keypoint/wider_naive_hrnet_18_256x192_coco.pdparams) | [config](./lite_hrnet/wider_naive_hrnet_18_256x192_coco.yml)     |
| LiteHRNet-18                   | 256x192  |     66.5     | [lite_hrnet_18_256x192_coco.pdparams](https://bj.bcebos.com/v1/paddledet/models/keypoint/lite_hrnet_18_256x192_coco.pdparams) | [config](./lite_hrnet/lite_hrnet_18_256x192_coco.yml)     |
| LiteHRNet-18                   | 384x288  |     69.7     | [lite_hrnet_18_384x288_coco.pdparams](https://bj.bcebos.com/v1/paddledet/models/keypoint/lite_hrnet_18_384x288_coco.pdparams) | [config](./lite_hrnet/lite_hrnet_18_384x288_coco.yml)     |
| LiteHRNet-30                   | 256x192  |     69.4     | [lite_hrnet_30_256x192_coco.pdparams](https://bj.bcebos.com/v1/paddledet/models/keypoint/lite_hrnet_30_256x192_coco.pdparams) | [config](./lite_hrnet/lite_hrnet_30_256x192_coco.yml)     |
| LiteHRNet-30                   | 384x288  |     72.5     | [lite_hrnet_30_384x288_coco.pdparams](https://bj.bcebos.com/v1/paddledet/models/keypoint/lite_hrnet_30_384x288_coco.pdparams) | [config](./lite_hrnet/lite_hrnet_30_384x288_coco.yml)     |


备注： Top-Down模型测试AP结果基于GroundTruth标注框

MPII数据集
| 模型  | 输入尺寸 | PCKh(Mean) | PCKh(Mean@0.1) |                           模型下载                           | 配置文件                                     |
| :---- | -------- | :--------: | :------------: | :----------------------------------------------------------: | -------------------------------------------- |
| HRNet-w32 | 256x256  |    90.6    |      38.5      | [hrnet_w32_256x256_mpii.pdparams](https://paddledet.bj.bcebos.com/models/keypoint/hrnet_w32_256x256_mpii.pdparams) | [config](./hrnet/hrnet_w32_256x256_mpii.yml) |


我们同时推出了针对移动端设备优化的实时关键点检测模型[PP-TinyPose](./tiny_pose/README.md), 欢迎体验。

## 快速开始

### 1、环境安装

​    请参考PaddleDetection [安装文档](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/docs/tutorials/INSTALL_cn.md)正确安装PaddlePaddle和PaddleDetection即可。

### 2、数据准备

​    目前KeyPoint模型支持[COCO](https://cocodataset.org/#keypoints-2017)数据集和[MPII](http://human-pose.mpi-inf.mpg.de/#overview)数据集，数据集的准备方式请参考[关键点数据准备](../../docs/tutorials/PrepareKeypointDataSet_cn.md)。

​    关于config配置文件内容说明请参考[关键点配置文件说明](../../docs/tutorials/KeyPointConfigGuide_cn.md)。


  - 请注意，Top-Down方案使用检测框测试时，需要通过检测模型生成bbox.json文件。COCO val2017的检测结果可以参考[Detector having human AP of 56.4 on COCO val2017 dataset](https://paddledet.bj.bcebos.com/data/bbox.json)，下载后放在根目录（PaddleDetection）下，然后修改config配置文件中`use_gt_bbox: False`后生效。然后正常执行测试命令即可。


### 3、训练与测试

​    **单卡训练：**

```shell
#COCO DataSet
CUDA_VISIBLE_DEVICES=0 python3 tools/train.py -c configs/keypoint/higherhrnet/higherhrnet_hrnet_w32_512.yml

#MPII DataSet
CUDA_VISIBLE_DEVICES=0 python3 tools/train.py -c configs/keypoint/hrnet/hrnet_w32_256x256_mpii.yml
```

​    **多卡训练：**

```shell
#COCO DataSet
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m paddle.distributed.launch tools/train.py -c configs/keypoint/higherhrnet/higherhrnet_hrnet_w32_512.yml

#MPII DataSet
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m paddle.distributed.launch tools/train.py -c configs/keypoint/hrnet/hrnet_w32_256x256_mpii.yml
```

​    **模型评估：**

```shell
#COCO DataSet
CUDA_VISIBLE_DEVICES=0 python3 tools/eval.py -c configs/keypoint/higherhrnet/higherhrnet_hrnet_w32_512.yml

#MPII DataSet
CUDA_VISIBLE_DEVICES=0 python3 tools/eval.py -c configs/keypoint/hrnet/hrnet_w32_256x256_mpii.yml

#当只需要保存评估预测的结果时，可以通过设置save_prediction_only参数实现，评估预测结果默认保存在output/keypoints_results.json文件中
CUDA_VISIBLE_DEVICES=0 python3 tools/eval.py -c configs/keypoint/higherhrnet/higherhrnet_hrnet_w32_512.yml --save_prediction_only
```

​    **模型预测：**

​    注意：top-down模型只支持单人截图预测，如需使用多人图，请使用[联合部署推理]方式。或者使用bottom-up模型。

```shell
CUDA_VISIBLE_DEVICES=0 python3 tools/infer.py -c configs/keypoint/higherhrnet/higherhrnet_hrnet_w32_512.yml -o weights=./output/higherhrnet_hrnet_w32_512/model_final.pdparams --infer_dir=../images/ --draw_threshold=0.5 --save_txt=True
```

​    **部署预测：**

```shell
#导出模型
python tools/export_model.py -c configs/keypoint/higherhrnet/higherhrnet_hrnet_w32_512.yml -o weights=output/higherhrnet_hrnet_w32_512/model_final.pdparams

#部署推理
#keypoint top-down/bottom-up 单独推理，该模式下top-down模型只支持单人截图预测。
python deploy/python/keypoint_infer.py --model_dir=output_inference/higherhrnet_hrnet_w32_512/ --image_file=./demo/000000014439_640x640.jpg --device=gpu --threshold=0.5
python deploy/python/keypoint_infer.py --model_dir=output_inference/hrnet_w32_384x288/ --image_file=./demo/hrnet_demo.jpg --device=gpu --threshold=0.5

#detector 检测 + keypoint top-down模型联合部署（联合推理只支持top-down方式）
python deploy/python/det_keypoint_unite_infer.py --det_model_dir=output_inference/ppyolo_r50vd_dcn_2x_coco/ --keypoint_model_dir=output_inference/hrnet_w32_384x288/ --video_file=../video/xxx.mp4  --device=gpu
```

​    **与多目标跟踪模型FairMOT联合部署预测：**

```shell
#导出FairMOT跟踪模型
python tools/export_model.py -c configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608.pdparams

#用导出的跟踪和关键点模型Python联合预测
python deploy/python/mot_keypoint_unite_infer.py --mot_model_dir=output_inference/fairmot_dla34_30e_1088x608/ --keypoint_model_dir=output_inference/higherhrnet_hrnet_w32_512/ --video_file={your video name}.mp4 --device=GPU
```
**注意:**
 跟踪模型导出教程请参考`configs/mot/README.md`。

### 4、模型单独部署

​    我们提供了PaddleInference(服务器端)、PaddleLite(移动端)、第三方部署(MNN、OpenVino)支持。无需依赖训练代码，deploy文件夹下相应文件夹提供独立完整部署代码。
详见 [部署文档](../../deploy/README.md)介绍。

## Benchmark
我们给出了不同运行环境下的测试结果，供您在选用模型时参考。详细数据请见[Keypoint Inference Benchmark](./KeypointBenchmark.md)。

## 引用
```
@inproceedings{cheng2020bottom,
  title={HigherHRNet: Scale-Aware Representation Learning for Bottom-Up Human Pose Estimation},
  author={Bowen Cheng and Bin Xiao and Jingdong Wang and Honghui Shi and Thomas S. Huang and Lei Zhang},
  booktitle={CVPR},
  year={2020}
}

@inproceedings{SunXLW19,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Ke Sun and Bin Xiao and Dong Liu and Jingdong Wang},
  booktitle={CVPR},
  year={2019}
}

@article{wang2019deep,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Wang, Jingdong and Sun, Ke and Cheng, Tianheng and Jiang, Borui and Deng, Chaorui and Zhao, Yang and Liu, Dong and Mu, Yadong and Tan, Mingkui and Wang, Xinggang and Liu, Wenyu and Xiao, Bin},
  journal={TPAMI},
  year={2019}
}

@InProceedings{Zhang_2020_CVPR,
    author = {Zhang, Feng and Zhu, Xiatian and Dai, Hanbin and Ye, Mao and Zhu, Ce},
    title = {Distribution-Aware Coordinate Representation for Human Pose Estimation},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2020}
}

@inproceedings{Yulitehrnet21,
  title={Lite-HRNet: A Lightweight High-Resolution Network},
  author={Yu, Changqian and Xiao, Bin and Gao, Changxin and Yuan, Lu and Zhang, Lei and Sang, Nong and Wang, Jingdong},
  booktitle={CVPR},
  year={2021}
}
```
