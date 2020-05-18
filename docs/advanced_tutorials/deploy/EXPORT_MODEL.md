# 模型导出

训练得到一个满足要求的模型后，如果想要将该模型接入到C++预测库或者Serving服务，需要通过`tools/export_model.py`导出该模型。同时，会导出预测时使用的配置文件，路径与模型保存路径相同, 配置文件名为`infer_cfg.yml`。

**说明：**

- **输入部分：** 导出模型输入为网络输入图像，即原始图片经过预处理后的图像，具体预处理方式可参考配置文件中TestReader部分。各类检测模型的输入格式分别为：

| 模型系列名称 | 输入图像预处理方式 | 其他输入信息 |
| :---------: | ----------- | ---------- |
|  YOLO | 缩放至指定大小，归一化 | im\_size: 格式为[origin\_H, origin\_W], origin为原始图像 |
| SSD | 缩放至指定大小，归一化 | im\_shape: 格式为[origin\_H, origin\_W], origin为原始图像 |
| RCNN | 归一化，等比例缩放 | 1. im\_info： 格式为[input\_H, input\_W, scale]，input为输入图像，scale为```输入图像大小/原始图像大小```<br>  2. im\_shape：格式为[origin\_H, origin\_W, 1.], origin为原始图像 |
| RCNN+FPN | 归一化，等比例缩放，对图像填充0使得长宽均为32的倍数 | 1. im\_info： 格式为[input\_H, input\_W, scale]，input为输入图像，scale为```输入图像大小/原始图像大小```<br>  2. im\_shape：格式为[origin\_H, origin\_W, 1.], origin为原始图像 |
| RetinaNet | 归一化，等比例缩放，对图像填充0使得长宽均为128的倍数 | 1. im\_info： 格式为[input\_H, input\_W, scale]，input为输入图像，scale为```输入图像大小/原始图像大小```<br>  2. im\_shape：格式为[origin\_H, origin\_W, 1.], origin为原始图像 |
| Face   |  归一化  | im\_shape: 格式为[origin\_H, origin\_W], origin为原始图像  |


- **输出部分：** 导出模型输出统一为NMS的输出，形状为[N, 6], 其中N为预测框的个数，6为[class_id, score, x1, y1, x2, y2]。

- 模型导出不支持模型结构中包含```fluid.layers.py_func```的情况。


## 启动参数说明

|      FLAG      |      用途      |    默认值    |                 备注                      |
|:--------------:|:--------------:|:------------:|:-----------------------------------------:|
|       -c       |  指定配置文件  |     None     |                                           |
|  --output_dir  |  模型保存路径  |  `./output`  |  模型默认保存在`output/配置文件名/`路径下 |

## 使用示例

使用[训练/评估/推断](../../tutorials/GETTING_STARTED_cn.md)中训练得到的模型进行试用，脚本如下

```bash
# 导出FasterRCNN模型, 模型中data层默认的shape为3x800x1333
python tools/export_model.py -c configs/faster_rcnn_r50_1x.yml \
        --output_dir=./inference_model \
        -o weights=output/faster_rcnn_r50_1x/model_final
```

预测模型会导出到`inference_model/faster_rcnn_r50_1x`目录下，模型名和参数名分别为`__model__`和`__params__`。


## 设置导出模型的输入大小

使用Fluid-TensorRT进行预测时，由于<=TensorRT 5.1的版本仅支持定长输入，保存模型的`data`层的图片大小需要和实际输入图片大小一致。而Fluid C++预测引擎没有此限制。设置TestReader中的`image_shape`可以修改保存模型中的输入图片大小。示例如下:

```bash
# 导出FasterRCNN模型，输入是3x640x640
python tools/export_model.py -c configs/faster_rcnn_r50_1x.yml \
        --output_dir=./inference_model \
        -o weights=https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_1x.tar \
           TestReader.inputs_def.image_shape=[3,640,640]

# 导出YOLOv3模型，输入是3x320x320
python tools/export_model.py -c configs/yolov3_darknet.yml \
        --output_dir=./inference_model \
        -o weights=https://paddlemodels.bj.bcebos.com/object_detection/yolov3_darknet.tar \
           TestReader.inputs_def.image_shape=[3,320,320]

# 导出SSD模型，输入是3x300x300
python tools/export_model.py -c configs/ssd/ssd_mobilenet_v1_voc.yml \
        --output_dir=./inference_model \
        -o weights=https://paddlemodels.bj.bcebos.com/object_detection/ssd_mobilenet_v1_voc.tar \
           TestReader.inputs_def.image_shape=[3,300,300]
```

## Paddle Serving部署模型导出

如果您要将上述模型用于[Paddle Serving](https://github.com/PaddlePaddle/Serving)在线预估服务，操作如下

```bash
# 导出Serving模型需要安装paddle-serving-client
pip install paddle-serving-client
# 导出FasterRCNN模型, 模型中data层默认的shape为3x800x1333
python tools/export_serving_model.py -c configs/faster_rcnn_r50_1x.yml \
        --output_dir=./inference_model \
        -o weights=output/faster_rcnn_r50_1x/model_final
```

用于Serving的预测模型会导出到`inference_model/faster_rcnn_r50_1x`目录下，其中`serving_client`为客户端配置文件夹，`serving_server`为服务端配置文件夹，模型参数也在服务端配置文件夹中。

更多的信息详情参见  [使用Paddle Serving部署Faster RCNN模型](https://github.com/PaddlePaddle/Serving/tree/develop/python/examples/faster_rcnn_model)
