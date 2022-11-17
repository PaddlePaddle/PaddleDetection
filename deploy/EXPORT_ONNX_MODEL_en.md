# PaddleDetection Model Export as ONNX Format Tutorial

PaddleDetection Model support is saved in ONNX format and the list of current test support is as follows
| Model  | OP Version | NOTE |
| :---- | :----- | :--- |
| YOLOv3 |  11   |  Only batch=1 inferring is supported. Model export needs fixed shape |
| PP-YOLO | 11 | Only batch=1 inferring is supported. A MatrixNMS will be converted to an NMS with slightly different precision; Model export needs fixed shape |
| PP-YOLOv2 | 11 | Only batch=1 inferring is supported. MatrixNMS will be converted to NMS with slightly different precision; Model export needs fixed shape |
| PP-YOLO Tiny | 11 | Only batch=1 inferring is supported. Model export needs fixed shape |
| PP-YOLOE | 11 | Only batch=1 inferring is supported. Model export needs fixed shape |
| PP-PicoDet | 11 | Only batch=1 inferring is supported. Model export needs fixed shape |
| FCOS | 11 |Only batch=1 inferring is supported |
| PAFNet | 11 |- |
| TTFNet | 11 |-|
| SSD | 11 |Only batch=1 inferring is supported |
| PP-TinyPose | 11 | - |
| Faster RCNN | 16 | Only batch=1 inferring is supported, require paddle2onnx>=0.9.7|
| Mask RCNN | 16 | Only batch=1 inferring is supported, require paddle2onnx>=0.9.7|
| Cascade RCNN | 16 | Only batch=1 inferring is supported, require paddle2onnx>=0.9.7|
| Cascade Mask RCNN | 16 | Only batch=1 inferring is supported, require paddle2onnx>=0.9.7|


The function of saving ONNX is provided by [Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX). If there is feedback on related problems during conversion, Communicate with engineers in Paddle2ONNX's Github project via [ISSUE](https://github.com/PaddlePaddle/Paddle2ONNX/issues).

## Export Tutorial

### Step 1. Export the Paddle deployment model
Export procedure reference document[Tutorial on PaddleDetection deployment model export](./EXPORT_MODEL_en.md), for example:

- Models except RCNN series, take YOLOv3 as example
```
cd PaddleDetection
python tools/export_model.py -c configs/yolov3/yolov3_darknet53_270e_coco.yml \
                             -o weights=https://paddledet.bj.bcebos.com/models/yolov3_darknet53_270e_coco.pdparams \
                             TestReader.inputs_def.image_shape=[3,608,608] \
                             --output_dir inference_model
```
The derived models were saved in `inference_model/yolov3_darknet53_270e_coco/`, with the structure as follows
```
yolov3_darknet
  ├── infer_cfg.yml          # Model configuration file information
  ├── model.pdiparams        # Static diagram model parameters
  ├── model.pdiparams.info   # Parameter Information is not required
  └── model.pdmodel          # Static diagram model file
```
> check`TestReader.inputs_def.image_shape`, For YOLO series models, specify this parameter when exporting; otherwise, the conversion fails

- RCNN series models, take Faster RCNN as example

The conditional block needs to be removed in RCNN series when export ONNX model. Add `export_onnx=True` in command line
```
cd PaddleDetection
python tools/export_model.py -c configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml \
                             -o weights=https://paddledet.bj.bcebos.com/models/faster_rcnn_r50_fpn_1x_coco.pdparams \
                             export_onnx=True \
                             --output_dir inference_model
```
The derived models were saved in `inference_model/faster_rcnn_r50_fpn_1x_coco/`, with the structure as follows
```
faster_rcnn_r50_fpn_1x_coco
  ├── infer_cfg.yml          # Model configuration file information
  ├── model.pdiparams        # Static diagram model parameters
  ├── model.pdiparams.info   # Parameter Information is not required
  └── model.pdmodel          # Static diagram model file
```

### Step 2. Convert the deployment model to ONNX format
Install Paddle2ONNX (version 0.9.7 or higher)
```
pip install paddle2onnx
```
Use the following command to convert
```
# YOLOv3
paddle2onnx --model_dir inference_model/yolov3_darknet53_270e_coco \
            --model_filename model.pdmodel \
            --params_filename model.pdiparams \
            --opset_version 11 \
            --save_file yolov3.onnx

# Faster RCNN
paddle2onnx --model_dir inference_model/faster_rcnn_r50_fpn_1x_coco \
            --model_filename model.pdmodel \
            --params_filename model.pdiparams \
            --opset_version 16 \
            --save_file faster_rcnn.onnx
```
The transformed model is under the current path`yolov3.onnx` and `faster_rcnn.onnx`

### Step 3. Inference with onnxruntime
Install onnxruntime
```
pip install onnxruntime
```
Inference code examples are in [deploy/third_engine/onnx](./third_engine/onnx)

Use the following commands for inference：
```
# YOLOv3
python deploy/third_engine/onnx/infer.py
            --infer_cfg inference_model/yolov3_darknet53_270e_coco/infer_cfg.yml \
            --onnx_file yolov3.onnx \
            --image_file demo/000000014439.jpg

# Faster RCNN
python deploy/third_engine/onnx/infer.py
            --infer_cfg inference_model/faster_rcnn_r50_fpn_1x_coco/infer_cfg.yml \
            --onnx_file faster_rcnn.onnx \
            --image_file demo/000000014439.jpg
```
