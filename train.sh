export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 #windows和Mac下不需要执行该命令
python3.7 tools/train.py -c configs/gfl/gfl_r18vd_1x_coco.yml
# python3.7 tools/train.py -c configs/yolov3/yolov3_mobilenet_v1_270e_coco.yml --slim_config configs/slim/distill/yolov3_mobilenet_v1_coco_distill.yml 
# python3.7 tools/train.py -c configs/slim/distill/retinanet_resnet101_coco_distill.yml 

#python3.7 tools/train.py -c configs/gfl/gfl_ld_r18vd_1x_coco.yml --slim_config configs/slim/distill/gfl_ld_distill.yml



# export CUDA_VISIBLE_DEVICES=0,1,2,3 #windows和Mac下不需要执行该命令
# python3.7 -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/gfl/gfl_r18vd_1x_coco.yml

#python3.7 -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/gfl/gfl_ld_r18vd_1x_coco.yml --slim_config configs/slim/distill/gfl_ld_distill.yml