export FLAGS_allocator_strategy=auto_growth

job_name=jde_darknet53_30e_1088x608
config=configs/jde/${job_name}.yml
#log_dir=log_dir/${job_name}
# 1. training
#python3.7 -m paddle.distributed.launch --log_dir=${log_dir} --gpus 0,1,2,3 tools/train.py -c ${config} #&> ${job_name}.log &
#python3.7 -m paddle.distributed.launch --log_dir=${log_dir} --gpus 1,2,5,7 tools/train.py -c ${config} -o pretrain_weights=jde_paddle_1088x608.pdparams use_gpu=true --weight_type resume #&> ${job_name}.log &
#CUDA_VISIBLE_DEVICES=7 python3.7 tools/train.py -c ${config} -o use_gpu=true
#CUDA_VISIBLE_DEVICES=7 python3.7 tools/train.py -c ${config} -o pretrain_weights=jde_paddle_1088x608.pdparams use_gpu=true --weight_type resume

# 2. eval_det:
#CUDA_VISIBLE_DEVICES=5 python3.7 tools/eval.py -c ${config} -o use_gpu=true weights=output/jde_darknet53_30e_1088x608/0.pdparams
#CUDA_VISIBLE_DEVICES=5 python3.7 tools/eval.py -c ${config} -o use_gpu=true weights=jde_paddle_1088x608.pdparams

# 3. eval_emb
job_name=jde_darknet53_30e_1088x608_testemb
config=configs/jde/${job_name}.yml
#CUDA_VISIBLE_DEVICES=5 python3.7 tools/eval.py -c ${config} -o use_gpu=true weights=output/jde_darknet53_30e_1088x608/0.pdparams
#CUDA_VISIBLE_DEVICES=5 python3.7 tools/eval.py -c ${config} -o use_gpu=true weights=jde_paddle_1088x608.pdparams

# 4 eval_mot
job_name=jde_darknet53_30e_1088x608_track
config=configs/jde/${job_name}.yml
CUDA_VISIBLE_DEVICES=6 python3.7 tools/eval_mot.py -c ${config} -o use_gpu=true weights=jde_paddle_1088x608.pdparams --benchmark MOT16_debug

# 5 infer_mot
job_name=jde_darknet53_30e_1088x608_track
config=configs/jde/${job_name}.yml
#CUDA_VISIBLE_DEVICES=6 python3.7 tools/infer_mot.py -c ${config} -o use_gpu=true weights=jde_paddle_1088x608.pdparams --video_file test.mp4 --save_videos
