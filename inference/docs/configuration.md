# 预测部署方案配置文件说明
## 基本概念
预测部署方案的配置文件旨在给用户提供一个预测部署方案定制化接口。用户仅需理解该配置文件相关字段的含义，无需编写任何代码，即可定制化预测部署方案。为了更好地表达每个字段的含义，首先介绍配置文件中字段的类型。

### 字段类型
- **required**: 表明该字段必须显式定义，否则无法正常启动预测部署程序。
- **optional**: 表明该字段可忽略不写，预测部署系统会提供默认值，相关默认值将在下文介绍。

### 字段值类型
- **int**：表明该字段必须赋予整型类型的值。
- **string**：表明该字段必须赋予字符串类型的值。
- **list**：表明该字段必须赋予列表的值。
- **tuple**: 表明该字段必须赋予双元素元组的值。

## 字段介绍

```yaml
# 预测部署时所有配置字段需在DEPLOY字段下
DEPLOY: 
    # 类型：required int
    # 含义：是否使用GPU预测。 0:不使用  1:使用
    USE_GPU: 1
    # 类型：required string
    # 含义：模型和参数文件所在目录
    MODEL_PATH: "/path/to/model_directory"
    # 类型：required string
    # 含义：模型文件名
    MODEL_FILENAME: "__model__"
    # 类型：required string
    # 含义：参数文件名
    PARAMS_FILENAME: "__params__"
    # 类型：optional string
    # 含义：图像resize的类型。支持 UNPADDING 和 RANGE_SCALING模式。默认是UNPADDING模式。
    RESIZE_TYPE: "UNPADDING"
    # 类型：required tuple
    # 含义：当使用UNPADDING模式时，会将图像直接resize到该尺寸。
    EVAL_CROP_SIZE: (513, 513)
    # 类型：optional int
    # 含义：当使用RANGE_SCALING模式时，图像短边需要对齐该字段的值，长边会同比例
    # 的缩放，从而在保持图像长宽比例不变的情况下resize到新的尺寸。默认值为0。
    TARGET_SHORT_SIZE: 800
    # 类型：optional int
    # 含义: 当使用RANGE_SCALING模式时,长边不能缩放到比该字段的值大。默认值为0。
    RESIZE_MAX_SIZE: 1333
    # 类型：required list
    # 含义：图像进行归一化预处理时的均值
    MEAN: [104.008, 116.669, 122.675]
    # 类型：required list
    # 含义：图像进行归一化预处理时的方差
    STD: [1.0, 1.0, 1.0]
    # 类型：string
    # 含义：图片类型, rgb 或者 rgba
    IMAGE_TYPE: "rgb"
    # 类型：required int
    # 含义：图像分类类型数
    NUM_CLASSES: 2
    # 类型：required int
    # 含义：图片通道数
    CHANNELS : 3
    # 类型：required string
    # 含义：预处理方式，目前提供图像检测的通用预处理类DetectionPreProcessor.
    PRE_PROCESSOR: "DetectionPreProcessor"
    # 类型：required string
    # 含义：预测模式，支持 NATIVE 和 ANALYSIS
    PREDICTOR_MODE: "ANALYSIS"
    # 类型：required int
    # 含义：每次预测的 batch_size
    BATCH_SIZE : 3
    # 类型：optional int
    # 含义: 输入张量的个数。大部分模型不需要设置。 默认值为1.
    FEEDS_SIZE: 2
    # 类型: optional int
    # 含义: 将图像的边变为该字段的值的整数倍。在使用fpn模型时需要设为32。默认值为1。
    COARSEST_STRIDE: 32 
```