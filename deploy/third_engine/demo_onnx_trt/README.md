# PP-YOLOE 转ONNX-TRT教程

本教程内容为：使用PP-YOLOE模型导出转换为ONNX格式，并定制化修改网络，使用[EfficientNMS_TRT](https://github.com/NVIDIA/TensorRT/tree/main/plugin/efficientNMSPlugin) OP，
可成功运行在[TensorRT](https://github.com/NVIDIA/TensorRT)上，示例仅供参考

## 1. 环境依赖

CUDA 11.7 + [cuDNN 8.4.1](https://developer.nvidia.com/rdp/cudnn-archive) + [TensorRT 8.4.2.4](https://developer.nvidia.com/nvidia-tensorrt-8x-download)

```shell
pip install onnx
pip install paddle2onnx
pip install onnx-simplifier
pip install onnx-graphsurgeon --index-url https://pypi.ngc.nvidia.com
pip install pycuda
```

## 2. Paddle模型导出

```shell
python ./tools/export_model.py \
    -c configs/ppyoloe/ppyoloe_crn_l_300e_coco.yml \
    -o weights=ppyoloe_crn_l_300e_coco.pdparams \
    --output_dir=./inference_model
```

## 3. ONNX模型转换 + 定制化修改EfficientNMS_TRT

```shell
python ./deploy/third_engine/demo_onnx_trt/export.py \
    --model_dir inference_model/ppyoloe_crn_l_300e_coco \
    --model_filename model.pdmodel \
    --params_filename model.pdiparams \
    --save_file inference_model/ppyoloe_crn_l_300e_coco/model.onnx \
    --batch-size 6 \
    --img 640 \
    --conf-thres 0.35 \
    --iou-thres 0.45 \
    --max_boxes 100 \
    --opset 11
```

## 4. TensorRT Engine

```shell
trtexec \
    --onnx=inference_model/ppyoloe_crn_l_300e_coco/model.onnx \
    --saveEngine=inference_model/ppyoloe_crn_l_300e_coco/model.engine \
    --best \
    --fp16 # if export TensorRT fp16 model
```

**注意**：若运行报错，可尝试添加`--tacticSources=-cublasLt,+cublas`参数解决

## 5. 运行TensorRT推理

```shell
python ./deploy/third_engine/demo_onnx_trt/pycuda_infer.py \
    --engine inference_model/ppyoloe_crn_l_300e_coco/model.engine \
    --output_dir ./deploy/third_engine/demo_onnx_trt/output \
    --infer_dir your_test_image_dir \
    # or --infer_img test_image_path # if only infer one file
```
