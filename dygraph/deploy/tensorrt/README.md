# PaddleDetection 使用TensorRT预测部署教程
TensorRT是NVIDIA提出的用于统一模型部署的加速库，可以应用于V100、JETSON Xavier等硬件，它可以极大提高预测速度。

## 1、如何安装`Paddle TensorRT`
- Python安装包，请从[这里](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/Tables.html#whl-release) 下载带有tensorrt的安装包进行安装
- CPP预测库，请从[这里](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/05_inference_deployment/inference/build_and_install_lib_cn.html) 下载带有TensorRT编译的预测库
- 如果Python和CPP官网没有提供已编译好的安装包或预测库，请参考[源码安装](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/compile/linux-compile.html) 自行编译

## 2、准备导出的模型
模型导出具体请参考文档[PaddleDetection模型导出教程](../EXPORT_MODEL.md)。

## 3、TensorRT原理简单介绍
当训练完成后，导出预测部署格式的模型。当使用TensorRT引擎进行预测时，TensorRT会把可以融合的OP融合成一个大OP，从而减少数据拷贝时间，一个大OP也有助于计算加速。从而提高推理速度。
如下图是一个训练好的模型的结构，如下图所示：
![img](imgs/tensorrt_before.jpg)
TensorRT会把其中的卷积操作、加bias操作、Relu操作融合成一个大操作，结果如下图所示：
![img](imgs/tensorrt_after.jpg)
TensorRT还会把前后层进行融合，结果如下图所示：
![img](imgs/tensorrt_after1.jpg)
优化后的网络结构比原来网络结构，计算效率高，推理速度快。

## 4、PaddleDetection训练的模型如何使用TensorRT进行加速
### （1）Paddle预测库中如何使用TensorRT
在使用Paddle预测库构建预测器配置config时，打开TensorRT引擎就可以了：

```
config->EnableUseGpu(100, 0); // 初始化100M显存，使用GPU ID为0
config->GpuDeviceId();        // 返回正在使用的GPU ID
// 开启TensorRT预测，可提升GPU预测性能，需要使用带TensorRT的预测库
config->EnableTensorRtEngine(1 << 20             /*workspace_size*/,
                             batch_size        /*max_batch_size*/,
                             3                 /*min_subgraph_size*/,
                             AnalysisConfig::Precision::kFloat32 /*precision*/,
                             false             /*use_static*/,
                             false             /*use_calib_mode*/);

```

### （2）关于固定尺寸和动态尺寸
使用TensorRT预测时，需要注意输入TensorRT引擎的数据是否是固定尺寸。
#### <a>. 输入TensorRT引擎的数据是固定尺寸
TensorRT版本<=5时，使用TensorRT预测时，只支持固定尺寸输入。在导出模型时，需要原始config文件中设定`TestReader.inputs_def.image_shape`参数，导出模型后，这个参数会指定输入网络的第一个Tensor的尺寸。
如果实际输入数据尺寸和这个不一致，会报错。
或者在导出模型时指定`TestReader.inputs_def.image_shape=[3,640,640]`,具体请参考[PaddleDetection模型导出教程](../EXPORT_MODEL.md) 。

**注意：
（1）`TestReader.inputs_def.image_shape`并不一定都是输入TensorRT引擎的数据尺寸，`TestReader.inputs_def.image_shape`指定的是在`Pad`操作之前的图像数据尺寸。
（2）不使用TensorRT时，`TestReader.inputs_def.image_shape`这个参数可以不用设置
**

可以通过[visualdl](https://www.paddlepaddle.org.cn/paddle/visualdl/demo/graph) 打开`model.pdmodel`文件，查看输入的第一个Tensor尺寸是否是固定的，如果不指定，尺寸会用`？`表示，如下图所示：
![img](imgs/input_shape.png)

#### <b>. 输入TensorRT引擎的数据是动态尺寸
TensorRT版本>=6时，使用TensorRT预测时，可以支持动态尺寸输入。Paddle预测库关于动态尺寸输入请查看[Paddle CPP预测](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/05_inference_deployment/inference/native_infer.html) 的`SetTRTDynamicShapeInfo`函数说明。
`python\infer.py`设置动态尺寸输入参数说明：
- use_dynamic_shape 用于设定TensorRT的输入尺寸是否是动态尺寸，默认值：False
- trt_min_shape 用于设定TensorRT的输入图像的最小尺寸，默认值：1
- trt_max_shape 用于设定TensorRT的输入图像的最大尺寸，默认值：1280
- trt_opt_shape 用于设定TensorRT的输入图像的最优尺寸，默认值：640

**注意：`TensorRT`中动态尺寸设置是4维的，这里只设置输入图像的尺寸。**


## 5、常见问题QA
### （1）提示没有`tensorrt_op`
请检查是否使用带有TensorRT的Paddle Python包或预测库。

### （2）提示`op out of memory`
![img](imgs/error2.png)
检查GPU是否是别人也在使用，请尝试使用空闲GPU

### （3）提示`some trt inputs dynamic shape info not set`
![img](imgs/error3.png)

这是由于`TensorRT`会把网络结果划分成多个子图，我们只设置了输入数据的动态尺寸，划分的其他子图的输入并未设置动态尺寸。有两个解决方法：
#### <a>通过增大`min_subgraph_size`，跳过对这些子图的优化。根据提示，设置min_subgraph_size大于并未设置动态尺寸输入的子图中OP个数即可。
`min_subgraph_size`的意思是，在加载TensorRT引擎的时候，大于`min_subgraph_size`的OP才会被优化，并且这些OP是连续的且是TensorRT可以优化的。

#### <b>找到子图的这些输入，按照上面方式也设置子图的输入动态尺寸。
