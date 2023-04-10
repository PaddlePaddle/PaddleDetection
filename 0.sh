export FLAGS_allocator_strategy=auto_growth
name=b
# name=l
# name=h
model_name=sam
job_name=sam_vit_${name}_coco
config=configs/${model_name}/${job_name}.yml
log_dir=log_dir/${job_name}
weights=../sam_vit_b_01ec64.pdparams
# weights=../sam_vit_l_0b3195.pdparams
# weights=../sam_vit_h_4b8939.pdparams
#weights=../sam_vit_${name}.pdparams

# 1. training
#CUDA_VISIBLE_DEVICES=3 python3.7 tools/train.py -c ${config} --amp #-r ${weights}
#python3.7 -m paddle.distributed.launch --log_dir=${log_dir} --gpus 1,2,3,4,5,6 tools/train.py -c ${config} --eval # --amp

# 2.eval
#CUDA_VISIBLE_DEVICES=1 python3.7 tools/eval.py -c ${config} -o weights=${weights} #--amp

# 3. tools infer
CUDA_VISIBLE_DEVICES=1 python3.7 tools/infer.py -c ${config} -o weights=${weights} --infer_img=demo/000000014439.jpg --draw_threshold=0.2

# 4. export
#CUDA_VISIBLE_DEVICES=1 python3.7 tools/export_model.py -c ${config} -o weights=${weights} #exclude_nms=True trt=True

# 5. deploy infer
#CUDA_VISIBLE_DEVICES=1 python3.7 deploy/python/infer.py --model_dir=output_inference/${job_name} --image_file=demo/000000014439.jpg --device=GPU

# 6. deploy speed
#CUDA_VISIBLE_DEVICES=1 python3.7 deploy/python/infer.py --model_dir=output_inference/${job_name} --image_file=demo/000000014439.jpg --device=GPU --run_benchmark=True # --run_mode=trt_fp16

# 7. onnx export
#paddle2onnx --model_dir output_inference/${job_name} --model_filename model.pdmodel --params_filename model.pdiparams --opset_version 12 --save_file ${job_name}.onnx

