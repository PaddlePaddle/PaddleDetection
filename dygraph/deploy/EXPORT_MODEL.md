# PaddleDetection模型导出教程

## 模型导出
本章节介绍如何使用`tools/export_model.py`脚本导出模型。
### 导出模输入输出说明
- `PaddleDetection`中输入变量以及输入形状如下：
| 输入名称 | 输入形状 | 表示含义 |
| :---------: | ----------- | ---------- |
| image |  [None, 3, H, W] | 输入网络的图像，None表示batch维度，如果输入图像大小为变长，则H,W为None |
| im_shape | [None, 2] | 图像经过resize后的大小，表示为H,W, None表示batch维度 |
| scale_factor | [None, 2] | 输入图像大小比真实图像大小，表示为scale_y, scale_x |

**注意**具体预处理方式可参考配置文件中TestReader部分。


- PaddleDetection`中动转静导出模型输出统一为：

  - bbox, NMS的输出，形状为[N, 6], 其中N为预测框的个数，6为[class_id, score, x1, y1, x2, y2]。
  - bbox\_num, 每张图片对应预测框的个数，例如batch_size为2，输出为[N1, N2], 表示第一张图包含N1个预测框，第二张图包含N2个预测框，并且预测框的总个数和NMS输出的第一维N相同
  - mask，如果网络中包含mask，则会输出mask分支

**注意**模型动转静导出不支持模型结构中包含numpy相关操作的情况。


### 启动参数说明

|      FLAG      |      用途      |    默认值    |                 备注                      |
|:--------------:|:--------------:|:------------:|:-----------------------------------------:|
|       -c       |  指定配置文件  |     None     |                                           |
|  --output_dir  |  模型保存路径  |  `./output_inference`  |  模型默认保存在`output/配置文件名/`路径下 |

### 使用示例

使用训练得到的模型进行试用，脚本如下

```bash
# 导出YOLOv3模型
python tools/export_model.py -c configs/yolov3/yolov3_darknet53_270e_coco.yml --output_dir=./inference_model \
 -o weights=weights/yolov3_darknet53_270e_coco.pdparams
```

预测模型会导出到`inference_model/yolov3_darknet53_270e_coco`目录下，分别为`infer_cfg.yml`, `model.pdiparams`,  `model.pdiparams.info`, `model.pdmodel`。


### 设置导出模型的输入大小

使用Fluid-TensorRT进行预测时，由于<=TensorRT 5.1的版本仅支持定长输入，保存模型的`data`层的图片大小需要和实际输入图片大小一致。而Fluid C++预测引擎没有此限制。设置TestReader中的`image_shape`可以修改保存模型中的输入图片大小。示例如下:

```bash
# 导出YOLOv3模型，输入是3x640x640
python tools/export_model.py -c configs/yolov3/yolov3_darknet53_270e_coco.yml --output_dir=./inference_model \
 -o weights=weights/yolov3_darknet53_270e_coco.pdparams TestReader.inputs_def.image_shape=[3,640,640]
```
