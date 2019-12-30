# 模型预测

基于[模型导出](EXPORT_MODEL.md)保存inference_model，通过下列方法对保存模型进行预测，同时测试不同方法下的预测速度

需要基于develop分支编译TensorRT版本Paddle, 在编译命令中指定TensorRT路径：

```
cmake .. -DWITH_MKL=ON \
         -DWITH_GPU=ON \
         -DWITH_TESTING=ON \
         -DCMAKE_BUILD_TYPE=Release \
         -DCUDA_ARCH_NAME=Auto \
         -DCMAKE_INSTALL_PREFIX=`pwd`/output \
         -DON_INFER=ON \
         -DTENSORRT_ROOT=${PATH_TO_TensorRT} \

make -j20
make install
```

## 使用方式

```bash
# 使用Paddle预测库加载模型进行预测
export CUDA_VISIBLE_DEVICES=0
python tools/tensorrt_infer.py --model_path=output/yolov3_mobilenet_v1/ --infer_img=demo/000000570688.jpg --mode=trt_fp32 --visualize --arch=YOLO --min_subgraph_size=3

# 使用load_inference_model加载模型进行预测
export CUDA_VISIBLE_DEVICES=0
python tools/tensorrt_infer.py --model_path=output/yolov3_mobilenet_v1/ --infer_img=demo/000000570688.jpg --visualize --arch=YOLO --use_python_inference
```


主要参数说明：

1. mode: 基于Paddle预测库的预测模式，可选模式为：trt_fp32, trt_int8, fluid。
2. visualize: 是否保存可视化结果，默认保存路径为```output/```。
3. arch: 不同的网络结构，对应不同的输入，可选结构为：YOLO, SSD, RCNN, RetinaNet
4. min_subgraph_size: 最小TensorRT子图个数，默认为3，对于RCNN模型，需要设置30以上
5. use_python_inference: 是否使用load_inference_model预测
6. mean, std, target_shape: 数据预处理参数，归一化平均值，方差，输入尺寸。**输入尺寸大小须与模型导出时设置的尺寸保持一致**。 用户可以在```tools/tensorrt_infer.py```自定义预处理方式

更多参数可以通过`--help`查看
