[English](pphuman_attribute_en.md) | 简体中文

# PP-Human属性识别模块

行人属性识别在智慧社区，工业巡检，交通监控等方向都具有广泛应用，PP-Human中集成了属性识别模块，属性包含性别、年龄、帽子、眼镜、上衣下衣款式等。我们提供了预训练模型，用户可以直接下载使用。

| 任务                 | 算法 | 精度 | 预测速度(ms) |下载链接                                                                               |
|:---------------------|:---------:|:------:|:------:| :---------------------------------------------------------------------------------: |
| 行人检测/跟踪 |  PP-YOLOE | mAP: 56.3 <br> MOTA: 72.0 | 检测: 16.2ms <br> 跟踪：22.3ms |[下载链接](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip) |
| 行人属性高精度模型    |  PP-HGNet_small  |  mA: 95.4  | 单人 1.54ms | [下载链接](https://bj.bcebos.com/v1/paddledet/models/pipeline/PPHGNet_small_person_attribute_954_infer.zip) |
| 行人属性轻量级模型    |  PP-LCNet_x1_0  |  mA: 94.5  | 单人 0.54ms | [下载链接](https://bj.bcebos.com/v1/paddledet/models/pipeline/PPLCNet_x1_0_person_attribute_945_infer.zip) |
| 行人属性精度与速度均衡模型    |  PP-HGNet_tiny  |  mA: 95.2  | 单人 1.14ms | [下载链接](https://bj.bcebos.com/v1/paddledet/models/pipeline/PPHGNet_tiny_person_attribute_952_infer.zip) |


1. 检测/跟踪模型精度为[MOT17](https://motchallenge.net/)，[CrowdHuman](http://www.crowdhuman.org/)，[HIEVE](http://humaninevents.org/)和部分业务数据融合训练测试得到。
2. 行人属性分析精度为[PA100k](https://github.com/xh-liu/HydraPlus-Net#pa-100k-dataset)，[RAPv2](http://www.rapdataset.com/rapv2.html)，[PETA](http://mmlab.ie.cuhk.edu.hk/projects/PETA.html)和部分业务数据融合训练测试得到
3. 预测速度为V100 机器上使用TensorRT FP16时的速度, 该处测速速度为模型预测速度
4. 属性模型应用依赖跟踪模型结果，请在[跟踪模型页面](./pphuman_mot.md)下载跟踪模型，依自身需求选择高精或轻量级下载。
5. 模型下载后解压放置在PaddleDetection/output_inference/目录下。

## 使用方法

1. 从上表链接中下载模型并解压到```PaddleDetection/output_inference```路径下，并修改配置文件中模型路径，也可默认自动下载模型。设置```deploy/pipeline/config/infer_cfg_pphuman.yml```中`ATTR`的enable: True

`infer_cfg_pphuman.yml`中配置项说明：
```
ATTR:                                                                     #模块名称
  model_dir: output_inference/PPLCNet_x1_0_person_attribute_945_infer/    #模型路径
  batch_size: 8                                                           #推理最大batchsize
  enable: False                                                           #功能是否开启
```

2. 图片输入时，启动命令如下(更多命令参数说明，请参考[快速开始-参数说明](./PPHuman_QUICK_STARTED.md#41-参数说明))。
```python
#单张图片
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_pphuman.yml \
                                                   --image_file=test_image.jpg \
                                                   --device=gpu \

#图片文件夹
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_pphuman.yml \
                                                   --image_dir=images/ \
                                                   --device=gpu \

```
3. 视频输入时，启动命令如下
```python
#单个视频文件
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_pphuman.yml \
                                                   --video_file=test_video.mp4 \
                                                   --device=gpu \

#视频文件夹
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_pphuman.yml \
                                                   --video_dir=test_videos/ \
                                                   --device=gpu \
```

4. 若修改模型路径，有以下两种方式：

    - 方法一：```./deploy/pipeline/config/infer_cfg_pphuman.yml```下可以配置不同模型路径，属性识别模型修改ATTR字段下配置
    - 方法二：命令行中--config后面紧跟着增加`-o ATTR.model_dir`修改模型路径：
```python
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_pphuman.yml
                                                   -o ATTR.model_dir=output_inference/PPLCNet_x1_0_person_attribute_945_infer/\
                                                   --video_file=test_video.mp4 \
                                                   --device=gpu
```

测试效果如下：

<div width="600" align="center">
  <img src="https://user-images.githubusercontent.com/22989727/205597518-7a602bd5-e643-44a1-a4ca-03c9ffecd918.gif"/>
</div>

数据来源及版权归属：天覆科技，感谢提供并开源实际场景数据，仅限学术研究使用

## 方案说明

1. 目标检测/多目标跟踪获取图片/视频输入中的行人检测框，模型方案为PP-YOLOE，详细文档参考[PP-YOLOE](../../../configs/ppyoloe/README_cn.md)
2. 通过行人检测框的坐标在输入图像中截取每个行人
3. 使用属性识别分析每个行人对应属性，属性类型与PA100k数据集相同，具体属性列表如下：
```
- 性别：男、女
- 年龄：小于18、18-60、大于60
- 朝向：朝前、朝后、侧面
- 配饰：眼镜、帽子、无
- 正面持物：是、否
- 包：双肩包、单肩包、手提包
- 上衣风格：带条纹、带logo、带格子、拼接风格
- 下装风格：带条纹、带图案
- 短袖上衣：是、否
- 长袖上衣：是、否
- 长外套：是、否
- 长裤：是、否
- 短裤：是、否
- 短裙&裙子：是、否
- 穿靴：是、否
```

4. 属性识别模型方案为[StrongBaseline](https://arxiv.org/pdf/2107.03576.pdf)，模型结构更改为基于PP-HGNet、PP-LCNet的多分类网络结构，引入Weighted BCE loss提升模型效果。

## 参考文献
```
@article{jia2020rethinking,
  title={Rethinking of pedestrian attribute recognition: Realistic datasets with efficient method},
  author={Jia, Jian and Huang, Houjing and Yang, Wenjie and Chen, Xiaotang and Huang, Kaiqi},
  journal={arXiv preprint arXiv:2005.11909},
  year={2020}
}
```
