export FLAGS_allocator_strategy=auto_growth

job_name=jde_darknet53_30e_1088x608
config=configs/jde/${job_name}.yml
#log_dir=log_dir/${job_name}
#mkdir -p ${work_dir}
# 1. training
#python3.7 -m paddle.distributed.launch --log_dir=${log_dir} --gpu 0,1,2,3 tools/train.py -c ${config} --eval &> ${job_name}.log &
#CUDA_VISIBLE_DEVICES=7 python3.7 tools/train.py -c ${config} -o use_gpu=true --eval

#CUDA_VISIBLE_DEVICES=5 python3.7 tools/train.py -c ${config} -o pretrain_weights=jde_paddle_1088x608.pdparams use_gpu=true --weight_type resume

# 2. eval
#CUDA_VISIBLE_DEVICES=0 python3.7 tools/eval.py -c ${config} -o use_gpu=true weights=jde_paddle_1088x608.pdparams --json_eval ${log_dir}

# 3. infer
#CUDA_VISIBLE_DEVICES=3 python3.7 tools/infer.py -c ${config} -o use_gpu=true weights=${job_name}.pdparams --infer_img=../demo/000000014439.jpg --draw_threshold 0.5
CUDA_VISIBLE_DEVICES=5 python3.7 tools/infer.py -c ${config} -o use_gpu=true weights=jde_paddle_1088x608.pdparams --infer_img=../demo/000000570688.jpg --draw_threshold 0.5 #2>&1 | tee jde_infer_dy_print.txt

# 4 export
#CUDA_VISIBLE_DEVICES=5 python3.7 tools/export_model.py -c ${config} --output_dir=./inference_model -o weights=jde_paddle_1088x608.pdparams #TestReader.inputs_def.image_shape=[3,800,800]