export CUDA_VISIBLE_DEVICES=1 #windows和Mac下不需要执行该命令
#python3.7 tools/eval.py -c configs/gfl/gfl_r18vd_1x_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/gfl_r18vd_1x_coco.pdparams

#python3.7 tools/eval.py -c configs/gfl/gfl_r18vd_1x_coco.yml -o weights=output/gfl_r18vd_1x_coco/model_final.pdparams

python3.7 tools/eval.py -c configs/gfl/gfl_r18vd_1x_coco.yml -o weights=output/gfl_ld_distill/model_final.pdparams 

