# PaddleDetection SOPHGO部署示例

## 1. 支持模型列表

目前SOPHGO支持如下模型的部署
- [PP-YOLOE系列模型](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/ppyoloe)
- [PicoDet系列模型](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/picodet)
- [YOLOV8系列模型](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4)

## 2. 准备PP-YOLOE YOLOV8或者PicoDet部署模型以及转换模型

SOPHGO-TPU部署模型前需要将Paddle模型转换成bmodel模型，具体步骤如下:
- Paddle动态图模型转换为ONNX模型，请参考[PaddleDetection导出模型](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/deploy/EXPORT_MODEL.md).
- ONNX模型转换bmodel模型的过程，请参考[TPU-MLIR](https://github.com/sophgo/tpu-mlir)

## 3. 模型转换example

PP-YOLOE YOLOV8和PicoDet模型转换过程类似，下面以ppyoloe_crn_s_300e_coco为例子,教大家如何转换Paddle模型到SOPHGO-TPU模型

### 导出ONNX模型
```shell
#导出paddle模型
python tools/export_model.py -c configs/ppyoloe/ppyoloe_crn_s_300e_coco.yml --output_dir=output_inference -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_s_300e_coco.pdparams

#paddle模型转ONNX模型
paddle2onnx --model_dir ppyoloe_crn_s_300e_coco \
            --model_filename model.pdmodel \
            --params_filename model.pdiparams \
            --save_file ppyoloe_crn_s_300e_coco.onnx \
            --enable_dev_version True

#进入Paddle2ONNX文件夹，固定ONNX模型shape
python -m paddle2onnx.optimize --input_model ppyoloe_crn_s_300e_coco.onnx \
                                --output_model ppyoloe_crn_s_300e_coco.onnx \
                                --input_shape_dict "{'image':[1,3,640,640]}"

```
### 导出bmodel模型

以转化BM1684x的bmodel模型为例子，我们需要下载[TPU-MLIR](https://github.com/sophgo/tpu-mlir)工程，安装过程具体参见[TPU-MLIR文档](https://github.com/sophgo/tpu-mlir/blob/master/README.md)。
## 4.	安装
``` shell
docker pull sophgo/tpuc_dev:latest

# myname1234是一个示例，也可以设置其他名字
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest

source ./envsetup.sh
./build.sh
```

## 5.	ONNX模型转换为bmodel模型
``` shell
mkdir ppyoloe_crn_s_300e_coco && cd ppyoloe_crn_s_300e_coco

# 下载测试图片，并将图片转换为npz格式
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg

#使用python获得模型转换所需要的npz文件
im = cv2.imread(im)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
#[640 640]为ppyoloe_crn_s_300e_coco的输入大小
im_scale_y = 640 / float(im.shape[0])
im_scale_x = 640 / float(im.shape[1])
inputs = {}
inputs['image'] = np.array((im, )).astype('float32')
inputs['scale_factor'] = np.array([im_scale_y, im_scale_x]).astype('float32')
np.savez('inputs.npz', image = inputs['image'], scale_factor = inputs['scale_factor'])

#放入onnx模型文件ppyoloe_crn_s_300e_coco.onnx

mkdir workspace && cd workspace

# 将ONNX模型转换为mlir模型
model_transform.py \
    --model_name ppyoloe_crn_s_300e_coco \
    --model_def ../ppyoloe_crn_s_300e_coco.onnx \
    --input_shapes [[1,3,640,640],[1,2]] \
    --keep_aspect_ratio \
    --pixel_format rgb \
    --output_names p2o.Div.1,p2o.Concat.29 \
    --test_input ../inputs.npz \
    --test_result ppyoloe_crn_s_300e_coco_top_outputs.npz \
    --mlir ppyoloe_crn_s_300e_coco.mlir
```
## 6. 注意
**由于TPU-MLIR当前不支持后处理算法，所以需要查看后处理的输入作为网络的输出**  
具体方法为：output_names需要通过[NETRO](https://netron.app/) 查看，网页中打开需要转换的ONNX模型，搜索NonMaxSuppression节点  
查看INPUTS中boxes和scores的名字，这个两个名字就是我们所需的output_names  
例如使用Netron可视化后，可以得到如下图片  
![](https://user-images.githubusercontent.com/120167928/210939488-a37e6c8b-474c-4948-8362-2066ee7a2ecb.png)  
找到蓝色方框标记的NonMaxSuppression节点，可以看到红色方框标记的两个节点名称为p2o.Div.1,p2o.Concat.29

``` bash
# 将mlir模型转换为BM1684x的F32 bmodel模型
model_deploy.py \
  --mlir ppyoloe_crn_s_300e_coco.mlir \
  --quantize F32 \
  --chip bm1684x \
  --test_input ppyoloe_crn_s_300e_coco_in_f32.npz \
  --test_reference ppyoloe_crn_s_300e_coco_top_outputs.npz \
  --model ppyoloe_crn_s_300e_coco_1684x_f32.bmodel
```
最终获得可以在BM1684x上能够运行的bmodel模型ppyoloe_crn_s_300e_coco_1684x_f32.bmodel。如果需要进一步对模型进行加速，可以将ONNX模型转换为INT8 bmodel，具体步骤参见[TPU-MLIR文档](https://github.com/sophgo/tpu-mlir/blob/master/README.md)。

## 7. 详细的部署示例
- [Cpp部署](./cpp)
- [python部署](./python)
