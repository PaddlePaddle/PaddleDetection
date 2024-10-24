# 快速开始

>**说明：**
>* 飞桨低代码开发工具[PaddleX](https://github.com/PaddlePaddle/PaddleX/tree/release/3.0-beta1)，依托于PaddleDetection的先进技术，支持了目标检测领域的**低代码全流程**开发能力。通过低代码全流程开发，可实现简单且高效的模型使用、组合与定制。
>* PaddleX 致力于实现产线级别的模型训练、推理与部署。模型产线是指一系列预定义好的、针对特定AI任务的开发流程，其中包含能够独立完成某类任务的单模型（单功能模块）组合。本文档提供**目标检测相关产线**的快速使用，单功能模块的快速使用以及更多功能请参考[PaddleDetection低代码全流程开发](./overview.md)中相关章节。

### 🛠️ 安装

> ❗安装PaddleX前请先确保您有基础的**Python运行环境**。（注：当前支持Python 3.8 ～ Python 3.10下运行，更多Python版本适配中）。
* **安装PaddlePaddle**
```bash
# cpu
python -m pip install paddlepaddle==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# gpu，该命令仅适用于 CUDA 版本为 11.8 的机器环境
python -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# gpu，该命令仅适用于 CUDA 版本为 12.3 的机器环境
python -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/
```
> ❗ 更多飞桨 Wheel 版本请参考[飞桨官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)。

* **安装PaddleX**

```bash
pip install https://paddle-model-ecology.bj.bcebos.com/paddlex/whl/paddlex-3.0.0b1-py3-none-any.whl
```

> ❗ 更多安装方式参考[PaddleX安装教程](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/installation/installation.md)
### 💻 命令行使用

一行命令即可快速体验产线效果，统一的命令行格式为：

```bash
paddlex --pipeline [产线名称] --input [输入图片] --device [运行设备]
```

只需指定三个参数：
* `pipeline`：产线名称
* `input`：待处理的输入图片的本地路径或URL
* `device`: 使用的GPU序号（例如`gpu:0`表示使用第0块GPU），也可选择使用CPU（`cpu`）


以通用目标检测产线为例：
```bash
paddlex --pipeline object_detection --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_object_detection_002.png --device gpu:0
```
运行后，得到的结果为：

```
{'input_path': '/root/.paddlex/predict_input/general_object_detection_002.png', 'boxes': [{'cls_id': 49, 'label': 'orange', 'score': 0.8188097476959229, 'coordinate': [661, 93, 870, 305]}, {'cls_id': 47, 'label': 'apple', 'score': 0.7743489146232605, 'coordinate': [76, 274, 330, 520]}, {'cls_id': 47, 'label': 'apple', 'score': 0.7270504236221313, 'coordinate': [285, 94, 469, 297]}, {'cls_id': 46, 'label': 'banana', 'score': 0.5570532083511353, 'coordinate': [310, 361, 685, 712]}, {'cls_id': 47, 'label': 'apple', 'score': 0.5484835505485535, 'coordinate': [764, 285, 924, 440]}, {'cls_id': 47, 'label': 'apple', 'score': 0.5160726308822632, 'coordinate': [853, 169, 987, 303]}, {'cls_id': 60, 'label': 'dining table', 'score': 0.5142655968666077, 'coordinate': [0, 0, 1072, 720]}, {'cls_id': 47, 'label': 'apple', 'score': 0.5101479291915894, 'coordinate': [57, 23, 213, 176]}]}
```

![](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/object_detection/03.png)


其他产线的命令行使用，只需将`pipeline`参数调整为相应产线的名称。下面列出了每个产线对应的命令：


| 产线名称      | 使用命令                                                                                                                                                                                             |
|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 通用目标检测    | `paddlex --pipeline object_detection --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_object_detection_002.png --device gpu:0`                                   |
| 通用实例分割    | `paddlex --pipeline instance_segmentation --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_instance_segmentation_004.png --device gpu:0`                         |
| 小目标检测     | `paddlex --pipeline small_object_detection --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/small_object_detection.jpg --device gpu:0`                                              |



### 📝 Python脚本使用

使用 [测试文件](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/small_object_detection.jpg)，并将 `predict()` 替换为本地路径,几行代码即可完成产线的快速推理，以小目标检测产线为例：

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="small_object_detection")

output = pipeline.predict("small_object_detection.jpg")
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_img("./output/") ## 保存结果可视化图像
    res.save_to_json("./output/") ## 保存预测的结构化输出
```

运行后，得到的结果为：

```bash
{'input_path': '/root/.paddlex/predict_input/small_object_detection.jpg', 'boxes': [{'cls_id': 3, 'label': 'car', 'score': 0.9243856072425842, 'coordinate': [624, 638, 682, 741]}, {'cls_id': 3, 'label': 'car', 'score': 0.9206348061561584, 'coordinate': [242, 561, 356, 613]}, {'cls_id': 3, 'label': 'car', 'score': 0.9194547533988953, 'coordinate': [670, 367, 705, 400]}, {'cls_id': 3, 'label': 'car', 'score': 0.9162291288375854, 'coordinate': [459, 259, 523, 283]}, {'cls_id': 4, 'label': 'van', 'score': 0.9075379371643066, 'coordinate': [467, 213, 498, 242]}, {'cls_id': 4, 'label': 'van', 'score': 0.9066920876502991, 'coordinate': [547, 351, 577, 397]}, {'cls_id': 3, 'label': 'car', 'score': 0.9041045308113098, 'coordinate': [502, 632, 562, 736]}, {'cls_id': 3, 'label': 'car', 'score': 0.8934890627861023, 'coordinate': [613, 383, 647, 427]}, {'cls_id': 3, 'label': 'car', 'score': 0.8803309202194214, 'coordinate': [640, 280, 671, 309]}, {'cls_id': 3, 'label': 'car', 'score': 0.8727016448974609, 'coordinate': [1199, 256, 1259, 281]}, {'cls_id': 3, 'label': 'car', 'score': 0.8705748915672302, 'coordinate': [534, 410, 570, 461]}, {'cls_id': 3, 'label': 'car', 'score': 0.8654043078422546, 'coordinate': [669, 248, 702, 271]}, {'cls_id': 3, 'label': 'car', 'score': 0.8555219769477844, 'coordinate': [525, 243, 550, 270]}, {'cls_id': 3, 'label': 'car', 'score': 0.8522038459777832, 'coordinate': [526, 220, 553, 243]}, {'cls_id': 3, 'label': 'car', 'score': 0.8392605185508728, 'coordinate': [557, 141, 575, 158]}, {'cls_id': 3, 'label': 'car', 'score': 0.8353804349899292, 'coordinate': [537, 120, 553, 133]}, {'cls_id': 3, 'label': 'car', 'score': 0.8322211503982544, 'coordinate': [585, 132, 603, 147]}, {'cls_id': 3, 'label': 'car', 'score': 0.8298957943916321, 'coordinate': [701, 283, 736, 313]}, {'cls_id': 3, 'label': 'car', 'score': 0.8217393159866333, 'coordinate': [885, 347, 943, 377]}, {'cls_id': 3, 'label': 'car', 'score': 0.820313572883606, 'coordinate': [493, 150, 511, 168]}, {'cls_id': 0, 'label': 'pedestrian', 'score': 0.8183429837226868, 'coordinate': [203, 701, 224, 743]}, {'cls_id': 0, 'label': 'pedestrian', 'score': 0.815082848072052, 'coordinate': [185, 710, 201, 744]}, {'cls_id': 6, 'label': 'tricycle', 'score': 0.7892289757728577, 'coordinate': [311, 371, 344, 407]}, {'cls_id': 6, 'label': 'tricycle', 'score': 0.7812919020652771, 'coordinate': [345, 380, 388, 405]}, {'cls_id': 0, 'label': 'pedestrian', 'score': 0.7748346328735352, 'coordinate': [295, 500, 309, 532]}, {'cls_id': 0, 'label': 'pedestrian', 'score': 0.7688500285148621, 'coordinate': [851, 436, 863, 466]}, {'cls_id': 3, 'label': 'car', 'score': 0.7466475367546082, 'coordinate': [565, 114, 580, 128]}, {'cls_id': 3, 'label': 'car', 'score': 0.7156463265419006, 'coordinate': [483, 66, 495, 78]}, {'cls_id': 3, 'label': 'car', 'score': 0.704211950302124, 'coordinate': [607, 138, 642, 152]}, {'cls_id': 3, 'label': 'car', 'score': 0.7021926045417786, 'coordinate': [505, 72, 518, 83]}, {'cls_id': 0, 'label': 'pedestrian', 'score': 0.6897469162940979, 'coordinate': [802, 460, 815, 488]}, {'cls_id': 3, 'label': 'car', 'score': 0.671891450881958, 'coordinate': [574, 123, 593, 136]}, {'cls_id': 9, 'label': 'motorcycle', 'score': 0.6712754368782043, 'coordinate': [445, 317, 472, 334]}, {'cls_id': 0, 'label': 'pedestrian', 'score': 0.6695684790611267, 'coordinate': [479, 309, 489, 332]}, {'cls_id': 3, 'label': 'car', 'score': 0.6273623704910278, 'coordinate': [654, 210, 677, 234]}, {'cls_id': 3, 'label': 'car', 'score': 0.6070230603218079, 'coordinate': [640, 166, 667, 185]}, {'cls_id': 3, 'label': 'car', 'score': 0.6064521670341492, 'coordinate': [461, 59, 476, 71]}, {'cls_id': 3, 'label': 'car', 'score': 0.5860581398010254, 'coordinate': [464, 87, 484, 100]}, {'cls_id': 9, 'label': 'motorcycle', 'score': 0.5792551636695862, 'coordinate': [390, 390, 419, 408]}, {'cls_id': 3, 'label': 'car', 'score': 0.5559225678443909, 'coordinate': [481, 125, 496, 140]}, {'cls_id': 0, 'label': 'pedestrian', 'score': 0.5531904697418213, 'coordinate': [869, 306, 880, 331]}, {'cls_id': 0, 'label': 'pedestrian', 'score': 0.5468509793281555, 'coordinate': [895, 294, 904, 319]}, {'cls_id': 3, 'label': 'car', 'score': 0.5451828241348267, 'coordinate': [505, 94, 518, 108]}, {'cls_id': 3, 'label': 'car', 'score': 0.5398445725440979, 'coordinate': [657, 188, 681, 208]}, {'cls_id': 4, 'label': 'van', 'score': 0.5318890810012817, 'coordinate': [518, 88, 534, 102]}, {'cls_id': 3, 'label': 'car', 'score': 0.5296525359153748, 'coordinate': [527, 71, 540, 81]}, {'cls_id': 6, 'label': 'tricycle', 'score': 0.5168400406837463, 'coordinate': [528, 320, 563, 346]}, {'cls_id': 3, 'label': 'car', 'score': 0.5088561177253723, 'coordinate': [511, 84, 530, 95]}, {'cls_id': 0, 'label': 'pedestrian', 'score': 0.502006471157074, 'coordinate': [841, 266, 850, 283]}]}
```

![](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/small_object_detection/02.png)

执行了如下几个步骤：

* `create_pipeline()` 实例化产线对象
* 传入图片并调用产线对象的`predict` 方法进行推理预测
* 对预测结果进行处理

其他产线的Python脚本使用，只需将`create_pipeline()`方法的`pipeline`参数调整为相应产线的名称。下面列出了每个产线对应的参数名称及详细的使用解释：

| 产线名称     | 对应参数                 | 详细说明 |
|----------|----------------------|------|
| 通用目标检测       | `object_detection` | [通用目标检测产线Python脚本使用说明](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/pipeline_usage/tutorials/cv_pipelines/object_detection.md) |
| 通用实例分割       | `instance_segmentation` | [通用实例分割产线Python脚本使用说明](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/pipeline_usage/tutorials/cv_pipelines/instance_segmentation.md) |
| 小目标检测         | `small_object_detection` | [小目标检测产线Python脚本使用说明](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/pipeline_usage/tutorials/cv_pipelines/small_object_detection.md) |

