
# PP-Vehicle违章停车识别模块

禁停区域违章停车识别在车辆应用场景中有着非常广泛的应用，借助AI的力量可以减轻人力投入，精准快速识别出违停车辆并进一步采取如广播驱离行为。PP-Vehicle中基于车辆跟踪模型、车牌检测模型和车牌识别模型实现了违章停车识别功能，具体模型信息如下：

| 任务                 | 算法 | 精度 | 预测速度(ms) |预测模型下载链接                                                                               |
|:---------------------|:---------:|:------:|:------:| :---------------------------------------------------------------------------------: |
| 车辆检测/跟踪 |  PP-YOLOE-l | mAP: 63.9 | - |[下载链接](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_ppvehicle.zip) |
| 车牌检测模型    |  ch_PP-OCRv3_det  |  hmean: 0.979  | - | [下载链接](https://bj.bcebos.com/v1/paddledet/models/pipeline/ch_PP-OCRv3_det_infer.tar.gz) |
| 车牌识别模型    |  ch_PP-OCRv3_rec  |  acc: 0.773  | - | [下载链接](https://bj.bcebos.com/v1/paddledet/models/pipeline/ch_PP-OCRv3_rec_infer.tar.gz) |

1. 跟踪模型使用PPVehicle数据集（整合了BDD100K-MOT和UA-DETRAC），是将BDD100K-MOT中的car, truck, bus, van和UA-DETRAC中的car, bus, van都合并为1类vehicle(1)后的数据集。
2. 车牌检测、识别模型使用PP-OCRv3模型在CCPD2019、CCPD2020混合车牌数据集上fine-tune得到。

## 使用方法

1. 用户可从上表链接中下载模型并解压到```PaddleDetection/output_inference```路径下，并修改配置文件中模型路径，也可默认自动下载模型。在```deploy/pipeline/config/examples/infer_cfg_illegal_parking.yml```中可手动设置三个模型的模型路径。

`infer_cfg_illegal_parking.yml`中配置项说明：
```
MOT:                                                                                             # 跟踪模块
  model_dir: https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_ppvehicle.zip  # 跟踪模型路径
  tracker_config: deploy/pipeline/config/tracker_config.yml                                      # 跟踪配置文件路径
  batch_size: 1                                                                                  # 跟踪batch size
  enable: True                                                                                   # 是否开启跟踪功能

VEHICLE_PLATE:                                                                                   # 车牌识别模块
  det_model_dir: https://bj.bcebos.com/v1/paddledet/models/pipeline/ch_PP-OCRv3_det_infer.tar.gz # 车牌检测模型路径
  det_limit_side_len: 480                                                                        # 检测模型单边输入尺寸
  det_limit_type: "max"                                                                          # 检测模型输入尺寸长短边选择，"max"表示长边
  rec_model_dir: https://bj.bcebos.com/v1/paddledet/models/pipeline/ch_PP-OCRv3_rec_infer.tar.gz # 车牌识别模型路径
  rec_image_shape: [3, 48, 320]                                                                  # 车牌识别模型输入尺寸
  rec_batch_num: 6                                                                               # 车牌识别batch size
  word_dict_path: deploy/pipeline/ppvehicle/rec_word_dict.txt                                    # OCR模型查询字典
  enable: True                                                                                   # 是否开启车牌识别功能
```

2. 输入视频，启动命令如下
```python
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/examples/infer_cfg_illegal_parking.yml \
                                                   --video_file=test_video.mp4 \
                                                   --device=gpu \
                                                   --draw_center_traj \
                                                   --illegal_parking_time=5 \
                                                   --region_type=custom \
                                                   --region_polygon 100 1000 1000 1000 900 1700 0 1700
```

参数说明如下：
- config：配置文件路径；
- video_file：测试视频路径；
- device：推理设备配置；
- draw_center_traj：画出车辆中心运动轨迹；
- illegal_parking_time：非法停车时间，单位为秒；
- region_type：非法停车区域类型，custom表示自定义；
- region_polygon：自定义非法停车多边形，至少为3个点。

**注意:**
 - 违章停车的测试视频必须是静止摄像头拍摄的，镜头不能抖动或移动。
 - 判断车辆是否在违停区域内是**以车辆的中心点**作为参考，车辆擦边而过等场景不算作违章停车。
 - `--region_polygon`表示用户自定义区域的多边形的点坐标序列，每两个为一对点坐标(x,y)，**按顺时针顺序**连成一个**封闭区域**，至少需要3对点也即6个整数，默认值是`[]`，需要用户自行设置点坐标，如是四边形区域，坐标顺序是`左上、右上、右下、左下`。用户可以运行[此段代码](../../tools/get_video_info.py)获取所测视频的分辨率帧数，以及可以自定义画出自己想要的多边形区域的可视化并自己调整。
 自定义多边形区域的可视化代码运行如下：
 ```python
 python get_video_info.py --video_file=demo.mp4 --region_polygon 200 200 400 200 300 400 100 400
 ```
 快速画出想要的区域的小技巧：先任意取点得到图片，用画图工具打开，鼠标放到想要的区域点上会显示出坐标，记录下来并取整，作为这段可视化代码的region_polygon参数，并再次运行可视化，微调点坐标参数直至满意。


3. 若修改模型路径，有以下两种方式：

    - 方法一：```./deploy/pipeline/config/examples/infer_cfg_illegal_parking.yml```下可以配置不同模型路径；
    - 方法二：命令行中--config配置项后面增加`-o VEHICLE_PLATE.det_model_dir=[YOUR_DETMODEL_PATH] VEHICLE_PLATE.rec_model_dir=[YOUR_RECMODEL_PATH]`修改模型路径。


测试效果如下：

<div width="600" align="center">
  <img src="https://user-images.githubusercontent.com/22989727/205598624-bcf5165c-990c-4fe4-8cde-eb1d45298d8f.gif"/>
</div>

可视化视频中左上角num后面的数值表示当前帧中车辆的数目；Total count表示画面中出现的车辆的总数，包括出现又消失的车辆。

## 方案说明

1. 目标检测/多目标跟踪获取图片/视频输入中的车辆检测框，模型方案为PP-YOLOE，详细文档参考[PP-YOLOE](../../../configs/ppyoloe/README_cn.md)
2. 基于跟踪算法获取每辆车的轨迹，如果车辆中心在违停区域内且在指定时间内未发生移动，则视为违章停车；
3. 使用车牌识别模型得到违章停车车牌并可视化。

## 参考资料

1. PaddeDetection特色检测模型[PP-YOLOE](../../../../configs/ppyoloe)。
2. Paddle字符识别模型库[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)。
