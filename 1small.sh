export FLAGS_allocator_strategy=auto_growth
model_type=ppyoloe
job_name=ppyoloe_plus_crn_s_80e_coco
job_name_tea=ppyoloe_plus_crn_m_80e_coco_distill
config=configs/${model_type}/${job_name}.yml
slim_config=configs/${model_type}/${job_name_tea}.yml
log_dir=log_dir/${job_name}
weights=output/${job_name}/model_final.pdparams

# 1. training
#CUDA_VISIBLE_DEVICES=3 python3.7 tools/train.py -c ${config} #-r ${weights}
python3.7 -m paddle.distributed.launch --log_dir=${log_dir} --gpus 0,1,2,3,4,5,6,7 tools/train.py -c ${config} --slim_config ${slim_config} --eval # --amp

# 2. eval
#CUDA_VISIBLE_DEVICES=0 python3.7 tools/eval.py -c ${config} -o weights=https://paddledet.bj.bcebos.com/models/${job_name}.pdparams
#CUDA_VISIBLE_DEVICES=2 python3.7 tools/eval.py -c ${config} -o weights=${weights}
