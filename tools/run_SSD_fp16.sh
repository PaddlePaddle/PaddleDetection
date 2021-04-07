export CUDA_VISIBLE_DEVICES=4
export FLAGS_fraction_of_gpu_memory_to_use=0.8
python train.py -c /work/PaddleDetection/configs/ssd/ssd_vgg16_300_voc.yml \
-o use_gpu=ON 