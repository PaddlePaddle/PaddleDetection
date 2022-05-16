[English](attribute_en.md) | 简体中文

# PP-Human属性识别模块

行人属性识别在智慧社区，工业巡检，交通监控等方向都具有广泛应用，PP-Human中集成了属性识别模块，属性包含性别、年龄、帽子、眼镜、上衣下衣款式等。我们提供了预训练模型，用户可以直接下载使用。

| 任务                 | 算法 | 精度 | 预测速度(ms) |下载链接                                                                               |
|:---------------------|:---------:|:------:|:------:| :---------------------------------------------------------------------------------: |
| 行人检测/跟踪    |  PP-YOLOE | mAP: 56.3 <br> MOTA: 72.0 | 检测: 28ms <br> 跟踪：33.1ms | [下载链接](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip) |
| 行人属性分析    |  StrongBaseline  |  mA: 94.86  | 单人 2ms | [下载链接](https://bj.bcebos.com/v1/paddledet/models/pipeline/strongbaseline_r50_30e_pa100k.zip) |

1. 检测/跟踪模型精度为[MOT17](https://motchallenge.net/)，[CrowdHuman](http://www.crowdhuman.org/)，[HIEVE](http://humaninevents.org/)和部分业务数据融合训练测试得到
2. 行人属性分析精度为[PA100k](https://github.com/xh-liu/HydraPlus-Net#pa-100k-dataset)，[RAPv2](http://www.rapdataset.com/rapv2.html)，[PETA](http://mmlab.ie.cuhk.edu.hk/projects/PETA.html)和部分业务数据融合训练测试得到
3. 预测速度为T4 机器上使用TensorRT FP16时的速度, 速度包含数据预处理、模型预测、后处理全流程

## 使用方法

1. 从上表链接中下载模型并解压到```./output_inference```路径下
2. 图片输入时，启动命令如下
```python
python deploy/pphuman/pipeline.py --config deploy/pphuman/config/infer_cfg.yml \
                                                   --image_file=test_image.jpg \
                                                   --device=gpu \
                                                   --enable_attr=True
```
3. 视频输入时，启动命令如下
```python
python deploy/pphuman/pipeline.py --config deploy/pphuman/config/infer_cfg.yml \
                                                   --video_file=test_video.mp4 \
                                                   --device=gpu \
                                                   --enable_attr=True
```
4. 若修改模型路径，有以下两种方式：

    - ```./deploy/pphuman/config/infer_cfg.yml```下可以配置不同模型路径，属性识别模型修改ATTR字段下配置
    - **(推荐)**命令行中增加`--model_dir`修改模型路径：
```python
python deploy/pphuman/pipeline.py --config deploy/pphuman/config/infer_cfg.yml \
                                                   --video_file=test_video.mp4 \
                                                   --device=gpu \
                                                   --enable_attr=True \
                                                   --model_dir det=ppyoloe/
```

测试效果如下：

<div width="1000" align="center">
  <img src="./images/attribute.gif"/>
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

4. 属性识别模型方案为[StrongBaseline](https://arxiv.org/pdf/2107.03576.pdf)，模型结构为基于ResNet50的多分类网络结构，引入Weighted BCE loss和EMA提升模型效果。

## 参考文献
```
@article{jia2020rethinking,
  title={Rethinking of pedestrian attribute recognition: Realistic datasets with efficient method},
  author={Jia, Jian and Huang, Houjing and Yang, Wenjie and Chen, Xiaotang and Huang, Kaiqi},
  journal={arXiv preprint arXiv:2005.11909},
  year={2020}
}
```
