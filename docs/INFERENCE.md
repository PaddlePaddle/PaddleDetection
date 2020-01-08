# 模型预测

基于[模型导出](EXPORT_MODEL.md)保存inference_model，通过下列方法对保存模型进行预测，同时测试不同方法下的预测速度

## 使用方式

```bash
export CUDA_VISIBLE_DEVICES=0
python tools/cpp_infer.py --model_path=inference_model/faster_rcnn_r50_1x/ --config_path=tools/cpp_demo.yml --infer_img=demo/000000570688.jpg --visualize
```


主要参数说明：

1. model_path: inference_model保存路径
2. config_path: 数据预处理配置文件
3. infer_img: 待预测图片
4. visualize: 是否保存可视化结果，默认保存路径为```output/```。


更多参数可在```tools/cpp_demo.yml```中查看,

**注意**

1. 设置shape时必须保持与模型导出时shape大小一致
2. ```min_subgraph_size```的设置与模型arch相关，对部分arch需要调大该参数，一般设置为40适用于所有模型。适当的调小```min_subgraph_size```会对预测有小幅加速效果。例如YOLO中该参数可设置为3
## Paddle环境搭建

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

export LD_LIBRARY_PATH=${PATH_TO_TensorRT}/lib:$LD_LIBRARY_PATH
```
