export FLAGS_allocator_strategy=auto_growth
job_name=mcfairmot_dla34_30e_1088x608_visdrone
config=configs/mot/mcfairmot/${job_name}.yml
log_dir=log_dir/${job_name}

# 1. training
#CUDA_VISIBLE_DEVICES=7 python3.7 tools/train.py -c ${config}
#python3.7 -m paddle.distributed.launch --log_dir=${log_dir} --gpus 4,5,6,7 tools/train.py -c ${config} # &> ${job_name}.log &

# 2. eval
#CUDA_VISIBLE_DEVICES=0 python3.7 tools/eval_mot.py -c ${config} -o weights=https://paddledet.bj.bcebos.com/models/mot/${job_name}.pdparams
#CUDA_VISIBLE_DEVICES=7 python3.7 tools/eval_mot.py -c ${config} -o weights=output/${job_name}/model_final.pdparams
#CUDA_VISIBLE_DEVICES=7 python3.7 tools/eval_mot.py -c ${config} -o weights=new_mcfairmot.pdparams

# 3. tools infer
#CUDA_VISIBLE_DEVICES=0 python3.7 tools/infer_mot.py -c ${config} -o weights=output/${job_name}/model_final.pdparams --video_file=test_demo.mp4 --frame_rate=20 --save_videos
#CUDA_VISIBLE_DEVICES=0 python3.7 tools/infer_mot.py -c ${config} -o weights=output/${job_name}/model_final.pdparams --image_dir=demo --save_videos
#CUDA_VISIBLE_DEVICES=3 python3.7 tools/infer_mot.py -c ${config} -o weights=new_mcfairmot.pdparams --image_dir=dataset/mot/visdrone_mcmot/images/val/uav0000086_00000_v --save_images
#CUDA_VISIBLE_DEVICES=3 python3.7 tools/infer_mot.py -c ${config} -o weights=new_mcfairmot.pdparams --image_dir=dataset/mot/visdrone_mcmot/images/val/uav0000117_02622_v --save_images

# 4.export model
CUDA_VISIBLE_DEVICES=0 python3.7 tools/export_model.py -c ${config} -o weights=new_mcfairmot.pdparams
#CUDA_VISIBLE_DEVICES=0 python3.7 tools/export_model.py -c ${config} -o weights=output/${job_name}/model_final.pdparams

# 5. deploy infer
#CUDA_VISIBLE_DEVICES=0 python3.7 deploy/python/mot_jde_infer.py --model_dir=output_inference/${job_name} --video_file=test.mp4 --device=GPU --save_mot_txts --save_images

# 6. test deploy speed
#CUDA_VISIBLE_DEVICES=0 python3.7 deploy/python/mot_jde_infer.py --model_dir=output_inference/${job_name} --image_file=demo/000000014439.jpg --device=GPU --run_benchmark=True --trt_max_shape=1088 --trt_min_shape=608 --trt_opt_shape=608 --run_mode=trt_fp16
