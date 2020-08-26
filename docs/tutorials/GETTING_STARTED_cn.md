# PaddleDetection教程
目标检测就是在图像中找到感兴趣区域的位置和目标类别。PaddleDetection是基于PaddlePaddle的目标检测库。

本教程以路标数据集roadsign为例，使用YOLOv3算法详细说明了如何使用PaddleDetection训练一个目标检测模型，并对模型进行评估和部署。


## 一、环境配置
关于配置运行环境，请参考[安装指南](INSTALL_cn.md)  
PaddleDetection 和 PaddlePaddle 版本关系
**提示:**  
  1. PaddleDetection v0.4需要PaddlePaddle>=1.8.4  
  2. [AI Studio](https://aistudio.baidu.com/aistudio/index) 平台有预装好的开发环境，且提供免费的硬件资源，欢迎使用。  
     [AI Studio PaddleDetection快速开始演示](https://aistudio.baidu.com/aistudio/projectdetail/724548).  


## 二、准备数据
PaddleDetection默认支持[COCO](http://cocodataset.org)和[Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) 和[WIDER-FACE](http://shuoyang1213.me/WIDERFACE/) 数据源。  
同时还支持自定义数据源。数据准备详细内容请参考[如何准备训练数据](./PrepareDataSet.md) 。


    ```
    最终数据集文件组织结构为：

    ├── annotations
    │   ├── road0.xml
    │   ├── road1.xml
    │   ├── road10.xml
    │   |   ...
    ├── images
    │   ├── road0.jpg
    │   ├── road1.jpg
    │   ├── road2.jpg
    │   |   ...
    ├── label_list.txt
    ├── test.txt
    ├── train.txt
    └── valid.txt

    # label_list.txt 是类别名称列表，文件名必须是 label_list.txt
    cat label_list.txt

    crosswalk
    speedlimit
    stop
    trafficlight

    # train.txt 是训练数据集文件列表
    cat train.txt

    ./JPEGImages/road839.png ./Annotations/road839.xml
    ./JPEGImages/road363.png ./Annotations/road363.xml
    ...

    # valid.txt 是测试数据集文件列表
    cat valid.txt

    ./JPEGImages/road218.png ./Annotations/road218.xml
    ./JPEGImages/road681.png ./Annotations/road681.xml
    ```

    也可以下载准备好的数据，([下载链接](https://paddlemodels.bj.bcebos.com/object_detection/roadsign_voc.zip) ，解压到`dataset/`文件夹下重命名为`roadsign`即可。  
    准备好数据后，一般的我们要对数据有所了解，比如图像量，图像尺寸，每一类目标区域个数，目标区域大小等。如有必要，还要对数据进行清洗。  
    roadsign数据集统计:

    |    数据    |    图片数量    |
    | :--------: | :-----------: |
    |   train    |     701       |  
    |   valid    |     176       |  


**说明：**（1）用户数据，建议在训练前仔细检查数据，避免因数据标注格式错误或图像数据不完整造成训练过程中的crash  
（2）如果图像尺寸太大的话，在不限制读入数据尺寸情况下，占用内存较多，会造成内存/显存溢出，请合理设置batch_size，可从小到大尝试  

## 四、模型选择

PaddleDetection中提供了丰富的模型库，具体可在[模型库](../MODEL_ZOO_cn.md)中查看各个模型的指标，您可依据实际部署算力的情况，选择合适的模型:

- 当对速度要求较高时，推荐您使用[PP-YOLO](../../configs/ppyolo/README_cn.md)。
- 算力资源小时，推荐您使用[移动端模型](../../configs/mobile/README.md)，PaddleDetection中的移动端模型经过迭代优化，具有较高性价比。
- 算力资源强大时，推荐您使用[服务器端模型](../../configs/rcnn_enhance/README.md)，该模型是PaddleDetection提出的面向服务器端实用的目标检测方案。

同时也可以根据使用场景不同选择合适的模型：
- 当小物体检测时，推荐您使用两阶段检测模型，比如Faster RCNN系列模型，具体可在[模型库](../MODEL_ZOO_cn.md)中找到。
- 当在交通领域使用，如行人，车辆检测时，推荐您使用[特色垂类检测模型](../featured_model/CONTRIB_cn.md)。
- 当在竞赛中使用，推荐您使用竞赛冠军模型[CACascadeRCNN](../featured_model/champion_model/CACascadeRCNN.md)与[OIDV5_BASELINE_MODEL](../featured_model/champion_model/OIDV5_BASELINE_MODEL.md)。
- 当在人脸检测中使用，推荐您使用[人脸检测模型](../featured_model/FACE_DETECTION.md)。

同时也可以尝试PaddleDetection中开发的高性能[YOLOv3增强模型](../featured_model/YOLOv3_ENHANCEMENT.md)、[YOLOv4模型](../featured_model/YOLO_V4.md)与[Anchor Free模型](../featured_model/ANCHOR_FREE_DETECTION.md)等。

本教程选用YOLOv3作为训练模型。

## 五、训练

在[`configs/templates/`](../../configs/templates) 文件夹下，提供了常用算法的配置文件模板，您可以根据需要选择合适模板，每个配置文件模板都对各个参数的进行详细注释，方便您修改参数。

**[yolov3_mobilenet_v1_roadsign_voc_template.yml](configs/templates/yolov3_mobilenet_v1_roadsign_voc_template.yml)** 是以YOLOv3算法使用roadsign VOC格式数据集为例的配置模板。  

```
# 在yolov3 voc 数据格式的配置文件模板上修改，configs/yolov3_mobilenet_v1_roadsign_voc_template.yml中对参数有详细注释
cp configs/templates/yolov3_mobilenet_v1_roadsign_voc_template.yml configs/yolov3_mobilenet_v1_roadsign_voc.yml
```
将`configs/yolov3_mobilenet_v1_roadsign_voc.yml`中`weights`改为`output/yolov3_mobilenet_v1_roadsign_voc/best_model`  


配置文件中与用户自定义数据相关的参数如下：

|        参数        |          说明          |
| :----------------: | :-------------------: |
|      use_gpu       | 根据硬件选择是否使用GPU  |  
|     max_iters      |  训练轮数，每个iter会运行`batch_size * device_num`张图片  |  
|    num_classes     |        类别数量         |
|  pretrain_weights  |     于训练权重文件      |  
|      weights       |     best模型保存位置      |  
|      base_lr       |  跟batch_size配合设置   |
|     schedulers     | 跟batch_size、max_iters配合设置 |
| TrainReader/dataset_dir |  训练数据路径，以$(ppdet_root)为当前目录,如"dataset/xxx"|  
|  TrainReader/anno_path   |  训练数据列表和对应标签路径，$(dataset_dir)下的路径，如"train.txt"，详细参考上面数据格式说明|
|  use_default_label   | 是否使用数据集默认的类别名称列表。对于自定义数据集(非VOC、COCO)必须是`false`，且必须提供`label_list.txt`文件|

**配置文件中重要参数设置:**
配置文件设计思路请参考文档[配置模块设计与介绍](./config_doc/CONFIG_cn.md)  
如何新增模型请参考文档[新增模型算法](./MODEL_TECHNICAL_cn.md)  

- 1、batch_size  
    **特别注意的是，当使用PP-YOLO时，`use_fine_grained_loss=true`，`YOLOv3Loss`里`的batch_size`必须要和`TrainReader`的b`atch_size`保持一致**  
    batch_size根据硬件内存或显存大小设置，例如设置为1。  
    可参考[模型库](../MODEL_ZOO_cn.md)中查看各个模型的指标，可依据实际硬件情况，选择合适的batch_size  
    注意：在多线程时，需要内存较多。reader中`bufsize`是设置共享内存大小，也会影响内存的使用量。  

- 2、num_classes
    num_classes 数据中类别数量。注意在FasterRCNN中，需要将 `with_background=true 且 num_classes+1`  

- 3、dataset路径设置
    ```
    # 指定数据集格式
    !VOCDataSet
    dataset/xxx/
    ├── Annotations
    │   ├── xxx1.xml
    │   |   ...
    ├── JPEGImages
    │   ├── xxx1.png
    │   |   ...
    ├── label_list.txt (必须提供，且文件名称必须是label_list.txt )
    ├── train.txt (训练数据集文件列表, ./JPEGImages/xxx1.png ./Annotations/xxx1.xml)
    └── valid.txt (测试数据集文件列表)
    # 数据集相对路径
    dataset_dir: dataset/roadsign_voc
    # 标记文件名
    anno_path: train.txt
    # 是否包含背景类，若包含背景类，num_classes需要+1
    with_background: false
    # 对于VOC、COCO等数据集，可以不指定类别标签文件，use_default_label可以是true，用户自定义数据需要设置成false，且需要提供label_list.txt
    use_default_label: false
    ```

- 4、inputs_def 设置
    inputs_def 使用配置文件中的默认设置

- 5、max_iters  
    training schedule，1x表示训练约12epoch(1个epoch表示把所有训练数据都跑一遍)，在约第[8, 11]th epochs时改变学习率  
    max_iters为最大迭代次数，而一个iter会运行`batch_size * device_num`张图片，训练中的训练中的batch_size（即TrainReader.batch_size）例如设置为1  
    1x即约12 epoch，`max_iters = (12*total_train_image) / (batch_size * device_num)`，batch_size=1时，`701*12/(1*1)=8412`  
    **注意：batch_size（即TrainReader.batch_size）batch_size 和 YOLOv3Loss.batch_size 不一样，YOLOv3Loss.batch_size是仅当use_fine_grained_loss=true时计算Loss时使用，且需要设置成一样**  

- 6、预训练模型权重 pretrain_weights
    在训练用户自定义数据集时，对预训练模型进行选择性加载。pretrain_weights 可以是imagenet的预训练好的分类模型权重，也可以是在VOC或COCO数据集上的预训练的检测模型权重。
    可参考[检测模型库](../MODEL_ZOO_cn.md)中查看各个模型的指标，预训练模型配置文件和权重下载地址。Paddle分类模型请参考[PaddleModels](https://github.com/PaddlePaddle/models)  
    支持如下两种加载方式：

    (1)、直接加载预训练权重（**推荐方式**），通过配置pretrain_weights参数，模型中和预训练模型中对应参数形状不同的参数将自动被忽略。也可以通过`-o`参数指定:
    ```
    python tools/train.py -c configs/yolov3_mobilenet_v1_roadsign_voc.yml --eval \
                           -o pretrain_weights=https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1.tar
    ```

    (2)、通过设置 finetune_exclude_pretrained_params 参数显示指定训练过程中忽略参数的名字，任何参数名均可加入`finetune_exclude_pretrained_params`中，如果参数名通过通配符匹配方式能够匹配上`finetune_exclude_pretrained_params`设置的参数字段，则在模型加载时忽略该参数。
    也可以通过`-o`参数指定:
    ```
    python tools/train.py -c configs/yolov3_mobilenet_v1_roadsign_voc.yml --eval \
                           -o pretrain_weights=https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1.tar \
                              finetune_exclude_pretrained_params=['yolo_output']
    ```

    如果用户需要利用自己的数据进行finetune，模型结构不变，只需要忽略与类别数相关的参数，不同模型类型所对应的忽略参数字段如下表所示:  

    |      模型类型      |             忽略参数字段                  |
    | :----------------: | :---------------------------------------: |
    |     Faster RCNN    |          cls\_score, bbox\_pred           |
    |     Cascade RCNN   |          cls\_score, bbox\_pred           |
    |       Mask RCNN    | cls\_score, bbox\_pred, mask\_fcn\_logits |
    |  Cascade-Mask RCNN | cls\_score, bbox\_pred, mask\_fcn\_logits |
    |      RetinaNet     |           retnet\_cls\_pred\_fpn          |
    |        SSD         |                ^conv2d\_                  |
    |       YOLOv3       |              yolo\_output                 |


- 7、base_lr 配合 batch_size调整，参考 [学习率调整策略](../FAQ.md#faq%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98)  
    在使用imagenet的预训练模型时，训练时使用1张卡，单卡batch_size=1, base_lr=0.00125，base_lr随着`(batch_size * GPU卡数)` 等比例变化。  
    如果是检测预训练模型上fine-turn，学习率可以设置小一些，如设置为0.0001。

    **若loss出现nan，请将学习率再设置小一些试试。**  

- 8、sample_transforms 详细请参考[数据处理模块](./READER_cn.md).  
    sample_transforms为数据预处理、数据增强，是针对单张图像的操作。基于YOLOv3算法可以使用模板中默认配置，详细参考文档[数据处理模块](READER_cn.md)  
    `configs中的sample_transforms`，各个函数说明请参考`ppdet/data/transform/operators.py`
    - DecodeImage: 读图，可以选择将图片从BGR转到RGB，可以选择对一个batch中的图片做mixup增强
    - NormalizeBox: 将坐标归一化到[0,1]区间
    - ExpandImage: 以prob大小概率进行扩充，max_ratio为最大扩充倍数，mean为扩充区域的均值
    - RandomInterpImage: 随机选择一种插值方法 reisze image
    - RandomFlipImage: 以prob概率随机反转
    - NormalizeImage: 以mean、std归一化图像
    - PadBox: 如果 bboxes 数量小于 num_max_boxes，填充值为0的 box
    - BboxXYXY2XYWH: 坐标格式从 XYXY格式 转换成 XYWH 格式

- 9、batch_transforms
    batch_transforms是针对一个batch数据的操作。基于YOLOv3算法可以使用模板中默认配置，详细参考文档[数据处理模块](READER_cn.md)  
    `configs中的batch_transforms`，各个函数说明请参考`ppdet/data/transform/batch_operators.py`
    - RandomShape: 随机选择一个尺寸，对整个batch进行reszie。如果random_inter为True，会同时随机选择一个插值方法resize。

**注意：**  
(1) YOLOv3、PP-YOLO、FPN、RetinaNet预测导出时，输入图像尺寸必须是32的整数倍。  
(2) ResizeImage是把图像短边resize到目标尺寸，ExpandImage是  
(3) 注意预处理顺序，一般的顺序：读取图片 -> 图片的各类变形、变色操作(用到cv2, PIL库的) -> Flip/Permute等 -> 归一化操作。 注意，不合理的顺序会引起错误。

拷贝好的`configs/yolov3_mobilenet_v1_roadsign_voc.yml`已经适配roadsign数据集，可以直接开始训练。

```bash
# 如果有GPU硬件，先设置显卡
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 边训练边测试
python tools/train.py -c configs/yolov3_mobilenet_v1_roadsign_voc.yml --eval

# 若没有显卡，将配置文件中 use_gpu设置成 false，或者通过 -o 选项覆盖配置文件中的参数
python tools/train.py -c configs/yolov3_mobilenet_v1_roadsign_voc.yml --eval -o use_gpu=false

# 通过visualdl命令实时查看变化曲线
python tools/train.py -c configs/yolov3_mobilenet_v1_roadsign_voc.yml --eval -o use_gpu=false --use_vdl=True --vdl_log_dir=vdl_dir/scalar
# 通过visualdl实时查看变化曲线
visualdl --logdir vdl_dir/scalar/ --host <host_IP> --port <port_num>
```

PaddleDetection还支持分布式训练，使用`tools/train_multi_machine.py`可以启动分布式训练，目前同时支持单机单卡、单机多卡与多机多卡的训练过程，其训练参数与`tools/train.py`完全一致。
详细请参考文档[分布式训练](MULTI_MACHINE_TRAINING_cn.md).  


## 六、评估和预测
```bash
# 评估 默认使用训练过程中保存的best_model
python tools/eval.py -c configs/yolov3_mobilenet_v1_roadsign_voc.yml
'''
    inference time: xxx fps
    mAP(0.50, 11point) = xxx
'''

# 指定模型评估
python tools/eval.py -c configs/yolov3_mobilenet_v1_roadsign_voc.yml -o weights=output/yolov3_mobilenet_v1_roadsign_voc/best_model

# 保存评估结果：设置 save_prediction_only=true 或者通过 -o 参数设置，会在当前文件夹下生成 bbox.json
python tools/eval.py -c configs/yolov3_mobilenet_v1_roadsign_voc.yml -o save_prediction_only=true

# 预测
python tools/infer.py -c configs/yolov3_mobilenet_v1_roadsign_voc.yml --infer_img=demo/road554.png
```

预测结果如下图：  
![](../images/road554.png)


可以通过命令行`tools/train.py`设置参数，其中的`-o`参数可以设置修改配置文件里的参数。  
`tools/train.py`训练参数列表
以下列表可以通过`--help`查看

|         FLAG             |     支持脚本    |        用途        |      默认值       |         备注         |
| :----------------------: | :------------: | :---------------: | :--------------: | :-----------------: |
|          -c              |      ALL       |  指定配置文件  |  None  |  **配置模块说明请参考[配置模块](../tutorials/config_doc/CONFIG_cn.md)** |
|          -o              |      ALL       |  设置配置文件里的参数内容  |  None  |  使用-o配置相较于-c选择的配置文件具有更高的优先级。例如：`-o use_gpu=False max_iter=10000`  |  
|   -r/--resume_checkpoint |     train      |  从某一检查点恢复训练  |  None  |  `-r output/faster_rcnn_r50_1x/10000`  |
|        --eval            |     train      |  是否边训练边测试  |  False  |    |
|      --output_eval       |     train/eval |  编辑评测保存json路径  |  当前路径  |  `--output_eval ./json_result`  |
|       --fp16             |     train      |  是否使用混合精度训练模式  |  False  |  需使用GPU训练  |
|       --loss_scale       |     train      |  设置混合精度训练模式中损失值的缩放比例  |  8.0  |  需先开启`--fp16`后使用  |  
|       --json_eval        |       eval     |  是否通过已存在的bbox.json或者mask.json进行评估  |  False  |  json文件路径在`--output_eval`中设置  |
|       --output_dir       |      infer     |  输出预测后可视化文件  |  `./output`  |  `--output_dir output`  |
|    --draw_threshold      |      infer     |  可视化时分数阈值  |  0.5  |  `--draw_threshold 0.7`  |
|      --infer_dir         |       infer     |  用于预测的图片文件夹路径  |  None  |    |
|      --infer_img         |       infer     |  用于预测的图片路径  |  None  |  相较于`--infer_dir`具有更高优先级  |
|        --use_vdl          |   train/infer   |  是否使用[VisualDL](https://github.com/paddlepaddle/visualdl)记录数据，进而在VisualDL面板中显示  |  False  |  VisualDL需Python>=3.5    |
|        --vdl\_log_dir     |   train/infer   |  指定 VisualDL 记录数据的存储路径  |  train:`vdl_log_dir/scalar` infer: `vdl_log_dir/image`  |  VisualDL需Python>=3.5   |

**注意:**
**参数设置优先级， 命令行 -o 选项参数设置的优先级 > 配置文件参数设置优先级 > 配置文件中的__READER__.yml中的参数设置优先级，高优先级会覆盖低优先级的参数设置**



## 七、模型部署
模型训练完成后，若需要对模型量化、裁剪、蒸馏，请参考文档[模型压缩](slim)  
训练得到满足要求的模型后，首先导出预测模型和预测配置文件，然后通过以下几种方式进行部署，详细请参考文档[推理部署](docs/tutorials/deploy)
- [Python端推理部署(支持 Linux 和 Windows)](deploy/python)
- [C++端推理部署(支持 Linux 和 Windows)](deploy/cpp)
- [PaddeServing](https://github.com/PaddlePaddle/Serving) :服务端使用PaddleServering部署，请参考[PaddeServing](https://github.com/PaddlePaddle/Serving) 部署成服务。
    PaddleServing yolov4 部署示例[PaddleServer yolov4 部署](https://github.com/PaddlePaddle/Serving/tree/develop/python/examples/yolov4)  
- [PaddleLite](https://github.com/PaddlePaddle/Paddle-Lite) :移动端使用Paddle-Lite部署，请参考[PaddleLite](https://github.com/PaddlePaddle/Paddle-Lite) 部署到移动端。
    Paddle-Lite 检测模型在android、iOS、armlinux上部署示例:[Paddle-Lite-Demo](https://github.com/PaddlePaddle/Paddle-Lite-Demo)
    注意：Lite框架目前支持的模型结构为PaddlePaddle深度学习框架产出的模型格式。可以使用[X2Paddle](https://github.com/PaddlePaddle/X2Paddle) 将其他框架模型转换成PaddlePaddle格式。再通过Paddle-Lite提供的opt工具转换成Paddle-Lite格式。
- [嵌入式端部署](../../configs/mobile/README.md)

部署服务时，首先需要导出预测的模型和预测配置文件，通过`tools/export_model.py`可以导出模型和名为`infer_cfg.yml`的配置文件。详细请参考[模型导出教程](docs/tutorials/deploy/EXPORT_MODEL_cn.md)

如果想测试模型的推理Benchmark，请参考文档[推理Benchmark](./deploy/BENCHMARK_INFER_cn.md)

这里以PaddleServering部署为例，可参考[PaddleServering yolov4部署例子](https://github.com/PaddlePaddle/Serving/tree/develop/python/examples/yolov4) 。

- (1) 导出Serving模型需要安装paddle-serving-client  
    `pip install paddle-serving-client`
- (2)导出模型
    训练得到一个满足要求的模型后，如果想要将该模型接入到C++预测库或者Serving服务，需要通过`tools/export_model.py`导出该模型。同时，会导出预测时使用的配置文件，路径与模型保存路径相同, 配置文件名为`infer_cfg.yml`。
    不同模型导出时，输入到模型中的数据预处理不同，详细请参考文档[模型导出](deploy/EXPORT_MODEL_cn.md)  

    ```
    python tools/export_serving_model.py -c configs/yolov3_mobilenet_v1_roadsign_voc.yml \
            --output_dir=./inference_model
    ```
    若要指定模型  
    ```
    python tools/export_serving_model.py -c configs/yolov3_mobilenet_v1_roadsign_voc.yml \
            --output_dir=./inference_model \
            -o weights=output/yolov3_mobilenet_v1_roadsign_voc/best_model
    ```
    以上命令会在./inference_model文件夹下生成一个 yolov3_mobilenet_v1_roadsign 文件夹
    ```
    inference_model
    │   ├── yolov3_mobilenet_v1_roadsign_voc.yml
    │   │   ├── infer_cfg.yml
    │   │   ├── serving_client
    │   │   │   ├── serving_client_conf.prototxt
    │   │   │   ├── serving_client_conf.stream.prototxt
    │   │   ├── serving_server
    │   │   │   ├── conv1_bn_mean
    │   │   │   ├── conv1_bn_offset
    │   │   │   ├── conv1_bn_scale
    │   │   │   ├── ...
    ```

    `serving_client`文件夹下`serving_client_conf.prototxt`详细说明了模型输入输出信息
    `serving_client_conf.prototxt`文件内容为：
    ```
    feed_var {
      name: "image"
      alias_name: "image"
      is_lod_tensor: false
      feed_type: 1
      shape: 3
      shape: 608
      shape: 608
    }
    feed_var {
      name: "im_size"
      alias_name: "im_size"
      is_lod_tensor: false
      feed_type: 2
      shape: 2
    }
    fetch_var {
      name: "multiclass_nms_0.tmp_0"
      alias_name: "multiclass_nms_0.tmp_0"
      is_lod_tensor: true
      fetch_type: 1
      shape: -1
    }
    ```
- (3) PaddleServer 预测，参考文档[PaddleServer yolov4 部署](https://github.com/PaddlePaddle/Serving/tree/develop/python/examples/yolov4)  
    ```
    cd inference_model/yolov3_mobilenet_v1_roadsign_voc/
    # GPU
    python -m paddle_serving_server_gpu.serve --model serving_server --port 9393 --gpu_ids 0
    # CPU
    python -m paddle_serving_server.serve --model serving_server --port 9393
    ```

    测试部署的服务
    ```
    # 进入到导出模型文件夹
    cd inference_model/yolov3_mobilenet_v1_roadsign_voc/
    # 将数据集对应的label_list.txt文件拷贝到当前文件夹下
    cp ../../dataset/roadsign_voc/label_list.txt .
    ```
    将 [test_client.py](https://github.com/PaddlePaddle/Serving/tree/develop/python/examples/yolov4/test_client.py) 拷贝到当前文件夹下  
    将代码中`yolov4_client/serving_client_conf.prototxt`路径改成
    `serving_client/serving_client_conf.prototxt`  
    将代码中`fetch=["save_infer_model/scale_0.tmp_0"])` 改成 `fetch=["multiclass_nms_0.tmp_0"])`

    文件组织结构如下
    ```
    inference_model
    │   ├── yolov3_mobilenet_v1_roadsign_voc.yml
    │   │   ├── label_list.txt
    │   │   ├── infer_cfg.yml
    │   │   ├── serving_client
    │   │   │   ├── serving_client_conf.prototxt
    │   │   │   ├── serving_client_conf.stream.prototxt
    │   │   ├── serving_server
    │   │   │   ├── conv1_bn_mean
    │   │   │   ├── conv1_bn_offset
    │   │   │   ├── conv1_bn_scale
    │   │   │   ├── ...
    │   │   ├── test_client.py
    ```

    测试
    ```
    # 测试代码 test_client.py 会自动创建output文件夹，并在output下生成`bbox.json`和`road554.png`两个文件
    python test_client.py ../../demo/road554.png
    ```

**如仍有疑惑，欢迎给我们提issue。**
