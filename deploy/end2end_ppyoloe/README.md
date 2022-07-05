# Export ONNX Model
## Download pretrain paddle models

* [ppyoloe-s](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_s_300e_coco.pdparams)
* [ppyoloe-m](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_m_300e_coco.pdparams)
* [ppyoloe-l](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams)
* [ppyoloe-x](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_x_300e_coco.pdparams)
* [ppyoloe-s-400e](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_s_400e_coco.pdparams)


## Export paddle model for deploying

```shell
python ./tools/export_model.py \
    -c configs/ppyoloe/ppyoloe_crn_s_300e_coco.yml \
    -o weights=ppyoloe_crn_s_300e_coco.pdparams \
    trt=True \
    exclude_nms=True \
    TestReader.inputs_def.image_shape=[3,640,640] \
    --output_dir ./

# if you want to try ppyoloe-s-400e model
python ./tools/export_model.py \
    -c configs/ppyoloe/ppyoloe_crn_s_400e_coco.yml \
    -o weights=ppyoloe_crn_s_400e_coco.pdparams \
    trt=True \
    exclude_nms=True \
    TestReader.inputs_def.image_shape=[3,640,640] \
    --output_dir ./
```

## Check requirements
```shell
pip install onnx>=1.10.0
pip install paddle2onnx
pip install onnx-simplifier
pip install onnx-graphsurgeon --index-url https://pypi.ngc.nvidia.com
# if use cuda-python infer, please install it
pip install cuda-python
# if use cupy infer, please install it
pip install cupy-cuda117 # cuda110-cuda117 are all available
```

## Export script
```shell
python ./deploy/end2end_ppyoloe/end2end.py \
    --model-dir ppyoloe_crn_s_300e_coco \
    --save-file ppyoloe_crn_s_300e_coco.onnx \
    --opset 11 \
    --batch-size 1 \
    --topk-all 100 \
    --iou-thres 0.6 \
    --conf-thres 0.4
# if you want to try ppyoloe-s-400e model
python ./deploy/end2end_ppyoloe/end2end.py \
    --model-dir ppyoloe_crn_s_400e_coco \
    --save-file ppyoloe_crn_s_400e_coco.onnx \
    --opset 11 \
    --batch-size 1 \
    --topk-all 100 \
    --iou-thres 0.6 \
    --conf-thres 0.4
```
#### Description of all arguments

- `--model-dir` : the path of ppyoloe export dir.
- `--save-file` : the path of export onnx.
- `--opset` : onnx opset version.
- `--img-size` : image size for exporting ppyoloe.
- `--batch-size` : batch size for exporting ppyoloe.
- `--topk-all` : topk objects for every image.
- `--iou-thres` : iou threshold for NMS algorithm.
- `--conf-thres` : confidence threshold for NMS algorithm.

### TensorRT backend (TensorRT version>= 8.0.0)
#### TensorRT engine export
``` shell
/path/to/trtexec \
    --onnx=ppyoloe_crn_s_300e_coco.onnx \
    --saveEngine=ppyoloe_crn_s_300e_coco.engine \
    --fp16 # if export TensorRT fp16 model
# if you want to try ppyoloe-s-400e model
/path/to/trtexec \
    --onnx=ppyoloe_crn_s_400e_coco.onnx \
    --saveEngine=ppyoloe_crn_s_400e_coco.engine \
    --fp16 # if export TensorRT fp16 model
```
#### TensorRT image infer

``` shell
# cuda-python infer script
python ./deploy/end2end_ppyoloe/cuda-python.py ppyoloe_crn_s_300e_coco.engine
# cupy infer script
python ./deploy/end2end_ppyoloe/cupy-python.py ppyoloe_crn_s_300e_coco.engine
# if you want to try ppyoloe-s-400e model
python ./deploy/end2end_ppyoloe/cuda-python.py ppyoloe_crn_s_400e_coco.engine
# or
python ./deploy/end2end_ppyoloe/cuda-python.py ppyoloe_crn_s_400e_coco.engine
```