# Export ONNX Model


## Export paddle model for deploying

```shell
python ./tools/export_model.py \
    -c configs/ppyoloe/ppyoloe_crn_l_300e_coco.yml \
    -o weights=ppyoloe_crn_l_300e_coco.pdparams \
    --output_dir=./inference_model
```

## Check requirements

```shell
pip install onnx
pip install paddle2onnx
pip install onnx-simplifier
pip install onnx-graphsurgeon --index-url https://pypi.ngc.nvidia.com
# if use pycuda infer, please install it
pip install pycuda
# if use cuda-python infer, please install it
pip install cuda-python
# if use cupy infer, please install it
pip install cupy-cuda117 # cuda110-cuda117 are all available
```

## Export script

```shell
# support PP-YOLOE(s,m,l,x) and PP-YOLOE+(s,m,l,x)
python ./deploy/end2end_ppyoloe/export.py \
    --model_dir inference_model/ppyoloe_crn_l_300e_coco \
    --model_filename model.pdmodel \
    --params_filename model.pdiparams \
    --save_file inference_model/ppyoloe_crn_l_300e_coco/model.onnx \
    --batch-size 6 \
    --img 640 \
    --conf-thres 0.35 \
    --iou-thres 0.45 \
    --max_boxes 100 \
    --opset 11
```

#### Description of all arguments

- `--model-dir` : Path of directory saved the input model.
- `--model_filename` : The input model file name.
- `--params_filename` : The parameters file name.
- `--save_file` : Path of directory to save the new exported model.
- `--batch-size` : total batch size for model.
- `--img` : inference size h,w.
- `--conf-thres` : confidence threshold.
- `--iou-thres` : NMS IoU threshold.
- `--opset` : opset version.
- `--max_boxes` : The maximum number of detections to output per image.

### TensorRT backend (TensorRT version>= 8.0.0)

#### TensorRT engine export

``` shell
trtexec \
    --onnx=inference_model/ppyoloe_crn_l_300e_coco/model.onnx \
    --saveEngine=inference_model/ppyoloe_crn_l_300e_coco/model.engine \
    --best \
    --fp16 # if export TensorRT fp16 model
```

#### TensorRT image infer

``` shell
# cuda-python infer script
python ./deploy/end2end_ppyoloe/cuda_infer.py \
    --engine inference_model/ppyoloe_crn_l_300e_coco/model.engine \
    --output_dir ./deploy/end2end_ppyoloe/output \
    --infer_dir your_test_image_dir \
    # or --infer_img test_image_path # if only infer one file
# cupy infer script
python ./deploy/end2end_ppyoloe/cupy_infer.py \
    --engine inference_model/ppyoloe_crn_l_300e_coco/model.engine \
    --output_dir ./deploy/end2end_ppyoloe/output \
    --infer_dir your_test_image_dir \
    # or --infer_img test_image_path # if only infer one file
# pycuda infer script
python ./deploy/end2end_ppyoloe/pycuda_infer.py \
    --engine inference_model/ppyoloe_crn_l_300e_coco/model.engine \
    --output_dir ./deploy/end2end_ppyoloe/output \
    --infer_dir your_test_image_dir \
    # or --infer_img test_image_path # if only infer one file
```