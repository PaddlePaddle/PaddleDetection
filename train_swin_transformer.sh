export PATH=/home/work/cuda-10.1/bin:$PATH
export LD_LIBRARY_PATH=/home/work/cuda-10.1/lib64$LD_LIBRARY_PATH

export LD_LIBRARY_PATH=/home/ssd5/wangyunhao02/other/env/nccl2_8.0/lib:$LD_LIBRARY_PATH 

# export CUDA_VISIBLE_DEVICES = 0,1
# python -m paddle.distributed.launch --gpus 0,1 tools/train.py  -c chen_faster_rcnn_studio-AdamW.yml -o find_unused_parameters=True --eval

export CUDA_VISIBLE_DEVICES = 0,1,2,3,4,5,6,7
python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py  -c swin_transformer_tiny.yml -o find_unused_parameters=True --eval
