[English](fight_recognition_en.md) | 简体中文

# PP-Human打架识别模块

随着监控摄像头部署覆盖范围越来越广，人工查看是否存在打架等异常行为耗时费力、效率低，AI+安防助理智慧安防。PP-Human中集成了打架识别模块，识别视频中是否存在打架行为。我们提供了预训练模型，用户可直接下载使用。

| 任务 | 算法 | 精度 | 预测速度(ms) | 模型权重 | 预测部署模型 |
| ---- | ---- | ---------- | ---- | ---- | ---------- |
|  打架识别 | PP-TSM | 准确率：89.06% | T4, 2s视频128ms | [下载链接](https://videotag.bj.bcebos.com/PaddleVideo-release2.3/ppTSM_fight.pdparams) | [下载链接](https://videotag.bj.bcebos.com/PaddleVideo-release2.3/ppTSM_fight.zip) |

打架识别模型基于6个公开数据集训练得到：Surveillance Camera Fight Dataset、A Dataset for Automatic Violence Detection in Videos、Hockey Fight Detection Dataset、Video Fight Detection Dataset、Real Life Violence Situations Dataset、UBI Abnormal Event Detection Dataset。

## 使用方法
1. 从上表链接中下载预测部署模型并解压到`./output_inference`路径下；
2. 修改解压后`ppTSM`文件夹中的文件名称为`model.pdiparams、model.pdiparams.info和model.pdmodel`；
3. 修改配置文件`deploy/pphuman/config/infer_cfg_pphuman.yml`中`VIDEO_ACTION`下的`enable`为`True`；
4. 输入视频，启动命令如下：
```
python deploy/pphuman/pipeline.py --config deploy/pphuman/config/infer_cfg.yml \
                                                   --video_file=test_video.mp4 \
                                                   --device=gpu
```

测试效果如下：

<div width="1000" align="center">
  <img src="./images/fight_demo.gif"/>
</div>

数据来源及版权归属：Surveillance Camera Fight Dataset。

## 方案说明

目前打架识别模型使用的是[PP-TSM](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/pp-tsm.md)，并在PP-TSM视频分类模型训练流程的基础上修改适配，完成模型训练。
