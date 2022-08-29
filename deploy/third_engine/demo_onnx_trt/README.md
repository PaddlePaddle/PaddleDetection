# PP-YOLOE 转ONNX-TRT教程

本教程内容为：使用PP-YOLOE模型导出转换为ONNX格式，并定制化修改网络，使用[EfficientNMS_TRT](https://github.com/NVIDIA/TensorRT/tree/main/plugin/efficientNMSPlugin) OP，
可成功运行在[TensorRT](https://github.com/NVIDIA/TensorRT)上，示例仅供参考

## 1. 环境依赖
CUDA 10.2 + [cudnn 8.2.1](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html) + [TensorRT 8.2](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-821/install-guide/index.htm)
```commandline
onnx
onnxruntime
paddle2onnx
```

## 2. Paddle模型导出
```commandline
python tools/export_model.py -c configs/ppyoloe/ppyoloe_crn_l_300e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams trt=True exclude_nms=True
```

## 3. ONNX模型转换 + 定制化修改EfficientNMS_TRT
```commandline
python deploy/third_engine/demo_onnx_trt/onnx_custom.py --onnx_file=output_inference/ppyoloe_crn_l_300e_coco/ppyoloe_crn_l_300e_coco.onnx --model_dir=output_inference/ppyoloe_crn_l_300e_coco/ --opset_version=11
```

## 4. TensorRT Engine
```commandline
trtexec --onnx=output_inference/ppyoloe_crn_l_300e_coco/ppyoloe_crn_l_300e_coco.onnx --saveEngine=ppyoloe_crn_l_300e_coco.engine
```
**注意**：若运行报错，可尝试添加`--tacticSources=-cublasLt,+cublas`参数解决

## 5. 运行TensorRT推理
```commandline
python deploy/third_engine/demo_onnx_trt/trt_infer.py --infer_cfg=output_inference/ppyoloe_crn_l_300e_coco/infer_cfg.yml --trt_engine=ppyoloe_crn_l_300e_coco.engine --image_file=demo/000000014439.jpg
```
