export CUDA_VISIBLE_DEVICES=0,1,2,3

# SingleCard
# python3.7 tools/train.py \
#         -c configs/picodet/picodet_s_416_coco_lcnet_fgd_student.yml \
#         --slim_config=configs/slim/distill/picodet_teacher_416_mgd_ssim_decay.yml --eval


# MultiCard
python3.7 -m paddle.distributed.launch --log_dir=./log --gpus '0,1,2,3' tools/train.py \
        -c configs/picodet/picodet_s_416_coco_lcnet_fgd_student.yml \
        --slim_config=configs/slim/distill/picodet_teacher_416_mgd_ssim_decay.yml --eval

