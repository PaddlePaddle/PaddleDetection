# PaddleDetection模型导出为ONNX格式教程

PaddleDetection模型支持保存为ONNX格式，目前测试支持的列表如下
| 模型  | OP版本 | 备注 |
| :---- | :----- | :--- |
| YOLOv3 |  11   |  仅支持batch=1推理；模型导出需固定shape |
| PPYOLO | 11 | 仅支持batch=1推理；MatrixNMS将被转成NMS，精度略有变化；模型导出需固定shape |
| PPYOLOv2 | 11 | 仅支持batch=1推理；MatrixNMS将被转换NMS，精度略有变化；模型导出需固定shape |
| PPYOLO-Tiny | 11 | 仅支持batch=1推理；模型导出需固定shape |
| FCOS | 11 |仅支持batch=1推理 |
| PAFNet | 11 |- |
| TTFNet | 11 |-|
| SSD | 11 |仅支持batch=1推理 |
| PicoDet | 11 |仅支持batch=1推理 |

保存ONNX的功能由[Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX)提供，如在转换中有相关问题反馈，可在Paddle2ONNX的Github项目中通过[ISSUE](https://github.com/PaddlePaddle/Paddle2ONNX/issues)与工程师交流。

## 导出教程

### 步骤一、导出PaddlePaddle部署模型
导出步骤参考文档[PaddleDetection部署模型导出教程](./EXPORT_MODEL.md), 以COCO数据集训练的YOLOv3为例，导出示例如下
```
cd PaddleDetection
python tools/export_model.py -c configs/yolov3/yolov3_darknet53_270e_coco.yml \
                             -o weights=https://paddledet.bj.bcebos.com/models/yolov3_darknet53_270e_coco.pdparams \
                             TestReader.inputs_def.image_shape=[3,608,608] \
                             --output_dir inference_model
```
导出后的模型保存在`inference_model/yolov3_darknet53_270e_coco/`目录中，结构如下
```
yolov3_darknet
  ├── infer_cfg.yml          # 模型配置文件信息
  ├── model.pdiparams        # 静态图模型参数
  ├── model.pdiparams.info   # 参数额外信息，一般无需关注
  └── model.pdmodel          # 静态图模型文件
```
> 注意导出时的参数`TestReader.inputs_def.image_shape`，对于YOLO系列模型注意导出时指定该参数，否则无法转换成功

### 步骤二、将部署模型转为ONNX格式
安装Paddle2ONNX（高于或等于0.6版本)
```
pip install paddle2onnx
```
使用如下命令转换
```
paddle2onnx --model_dir inference_model/yolov3_darknet53_270e_coco \
            --model_filename model.pdmodel \
            --params_filename model.pdiparams \
            --opset_version 11 \
            --save_file yolov3.onnx
```
转换后的模型即为在当前路径下的`yolov3.onnx`
