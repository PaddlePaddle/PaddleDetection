# PaddleDetection Predict deployment

PaddleDetection provides multiple deployment forms of Paddle Inference, Paddle Serving and Paddle-Lite, supports multiple platforms such as server, mobile and embedded, and provides a complete Python and C++ deployment solution

## PaddleDetection This section describes the supported deployment modes
| formalization    | language | Tutorial    | Equipment/Platform        |
| ---------------- | -------- | ----------- | ------------------------- |
| Paddle Inference | Python   | Has perfect | Linux(ARM\X86)、Windows   |
| Paddle Inference | C++      | Has perfect | Linux(ARM\X86)、Windows   |
| Paddle Serving   | Python   | Has perfect | Linux(ARM\X86)、Windows   |
| Paddle-Lite      | C++      | Has perfect | Android、IOS、FPGA、RK... |


## 1.Paddle Inference Deployment

### 1.1 The export model

Use the `tools/export_model.py` script to export the model and the configuration file used during deployment. The configuration file name is `infer_cfg.yml`. The model export script is as follows

```bash
# The YOLOv3 model is derived
python tools/export_model.py -c configs/yolov3/yolov3_mobilenet_v1_roadsign.yml -o weights=output/yolov3_mobilenet_v1_roadsign/best_model.pdparams
```
The prediction model will be exported to the `output_inference/yolov3_mobilenet_v1_roadsign` directory `infer_cfg.yml`, `model.pdiparams`,  `model.pdiparams.info`, `model.pdmodel`. For details on model export, please refer to the documentation [Tutorial on Paddle Detection MODEL EXPORT](EXPORT_MODEL_sh.md).

### 1.2 Use Paddle Inference to Make Predictions
* Python deployment supports `CPU`, `GPU` and `XPU` environments, Windows, Linux, and NV Jetson embedded devices. Reference Documentation [Python Deployment](python/README.md)
* C++ deployment supports `CPU`, `GPU` and `XPU` environments, Windows and Linux systems, and NV Jetson embedded devices. Reference documentation [C++ deployment](cpp/README.md)
* PaddleDetection supports TensorRT acceleration. Please refer to the documentation for [TensorRT Predictive Deployment Tutorial](TENSOR_RT.md)

**Attention:**  Paddle prediction library version requires >=2.1, and batch_size>1 only supports YOLOv3 and PP-YOLO.

##  2.PaddleServing Deployment
### 2.1 Export model

If you want to export the model in `PaddleServing` format, set `export_serving_model=True`:
```buildoutcfg
python tools/export_model.py -c configs/yolov3/yolov3_mobilenet_v1_roadsign.yml -o weights=output/yolov3_mobilenet_v1_roadsign/best_model.pdparams --export_serving_model=True
```
The prediction model will be exported to the `output_inference/yolov3_darknet53_270e_coco` directory `infer_cfg.yml`, `model.pdiparams`,  `model.pdiparams.info`, `model.pdmodel`, `serving_client/` and `serving_server/` folder.

For details on model export, please refer to the documentation [Tutorial on Paddle Detection MODEL EXPORT](EXPORT_MODEL_en.md).

### 2.2 Predictions are made using Paddle Serving
* [Install PaddleServing](https://github.com/PaddlePaddle/Serving/blob/develop/README.md#installation)
* [Use PaddleServing](./serving/README.md)


## 3. PaddleLite Deployment
- [Deploy the PaddleDetection model using PaddleLite](./lite/README.md)
- For details, please refer to [Paddle-Lite-Demo](https://github.com/PaddlePaddle/Paddle-Lite-Demo) deployment. For more information, please refer to [Paddle-Lite](https://github.com/PaddlePaddle/Paddle-Lite)


## 4. Benchmark Test
- Using the exported model, run the Benchmark batch test script:
```shell
sh deploy/benchmark/benchmark.sh {model_dir} {model_name}
```
**Attention** If it is a quantitative model, please use the `deploy/benchmark/benchmark_quant.sh` script.
- Export the test result log to Excel：
```
python deploy/benchmark/log_parser_excel.py --log_path=./output_pipeline --output_name=benchmark_excel.xlsx
```

## 5. FAQ
- 1、Can `Paddle 1.8.4` trained models be deployed with `Paddle2.0`?
  Paddle 2.0 is compatible with Paddle 1.8.4, so it is ok. However, some models (such as SOLOv2) use the new OP in Paddle 2.0, which is not allowed.

- 2、When compiling for Windows, the prediction library is compiled with VS2015, will it be a problem to choose VS2017 or VS2019?
  For compatibility issues with VS, please refer to: [C++ Visual Studio 2015, 2017 and 2019 binary compatibility](https://docs.microsoft.com/zh-cn/cpp/porting/binary-compat-2015-2017?view=msvc-160)

- 3、Does cuDNN 8.0.4 continuously predict memory leaks?
  QA tests show that cuDNN 8 series have memory leakage problems in continuous prediction, and cuDNN 8 performance is worse than cuDNN7. CUDA + cuDNN7.6.4 is recommended for deployment.
