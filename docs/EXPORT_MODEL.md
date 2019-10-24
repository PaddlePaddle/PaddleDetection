# 模型导出

训练得到一个满足要求的模型后，如果想要将该模型接入到C++预测库或者Serving服务，需要通过`tools/export_model.py`导出该模型。

## 启动参数说明

|      FLAG      |      用途      |    默认值    |                 备注                      |
|:--------------:|:--------------:|:------------:|:-----------------------------------------:|
|       -c       |  指定配置文件  |     None     |                                           |
|  --output_dir  |  模型保存路径  |  `./output`  |  模型默认保存在`output/配置文件名/`路径下 |

## 使用示例

使用[训练/评估/推断](GETTING_STARTED_cn.md)中训练得到的模型进行试用，脚本如下

```bash
# export model for RCNN
python tools/export_model.py -c configs/faster_rcnn_r50_1x.yml \
        --output_dir=./inference_model \
        -o weights=output/faster_rcnn_r50_1x/model_final \
           FasterRCNNTestFeed.image_shape=[3,800,1333]

# export model for YOLOv3
python tools/export_model.py -c configs/yolov3_darknet.yml \
        --output_dir=./inference_model \
        -o weights=output/yolov3_darknet/model_final \
           YoloTestFeed.image_shape=[3,800,1333]

# export model for SSD
python tools/export_model.py -c configs/ssd/ssd_mobilenet_v1_voc.yml \
        --output_dir=./inference_model \
        -o weights=output/ssd_mobilenet_v1_voc/model_final \
           SSDTestFeed.image_shape=[3,300,300]

```

- 预测模型会导出到`output/faster_rcnn_r50_1x`目录下，模型名和参数名分别为`__model__`和`__params__`。
- 通过`image_shape`修改保存模型中的图片大小。使用Fluid-TensorRT进行预测时，由于TensorRT仅支持定长输入，需要将输入图片大小与`image_shape`保持一致。
