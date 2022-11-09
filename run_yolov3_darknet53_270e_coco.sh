#!/bin/bash

# PR: https://github.com/PaddlePaddle/PaddleDetection/pull/6483
batch_size=${1:-8}
precision=${2:-"fp32"} # tf32, fp16_o1, fp16_o2
opt_level=${3:-1}
echo "batch_size=${batch_size}, precision=${precision}, opt_level=${opt_level}"

export CUDA_VISIBLE_DEVICES="4"

log_iter=10

# opt 1
if [ ${opt_level} -ge 1 ]; then
  use_shared_memory=True
else
  use_shared_memory=False
fi
  
# opt 2
if [ ${opt_level} -ge 2 ]; then
  if [ ${batch_size} -eq 16 ]; then
    num_workers=16
  else
    num_workers=2
  fi
else
  num_workers=2
fi
num_workers=16

# opt 3
if [ ${opt_level} -ge 3 ]; then
  export FLAGS_use_autotune=1
else
  export FLAGS_use_autotune=0
fi
#export GLOG_vmodule=switch_autotune=3

# opt 4
if [ ${opt_level} -ge 4 ]; then
  export FLAGS_conv_workspace_size_limit=4000 #MB
fi

# opt 5
if [ ${opt_level} -ge 5 ]; then
  export FLAGS_cudnn_batchnorm_spatial_persistent=True
fi

# opt 6
if [ ${opt_level} -ge 6 ]; then
  if [ "${precision}" = "fp16_o1" -o "${precision}" = "fp16_o2" ]; then
    data_format=NHWC
  else
    data_format=NCHW
  fi
else
  data_format=NCHW
fi

data_format=NCHW
export FLAGS_enable_eager_mode=1
if [ "${precision}" = "fp32" ]; then
  export NVIDIA_TF32_OVERRIDE=0
else
  unset NVIDIA_TF32_OVERRIDE
  if [ "${precision}" = "fp16_o1" ]; then
    amp_args="--amp"
    amp_level_args="amp_level=O1"
  elif [ "${precision}" = "fp16_o2" ]; then
    amp_args="--amp"
    amp_level_args="amp_level=O2"
  else
    amp_args=""
    amp_level_args=""
  fi
fi

#suffix=".profile"
#subdir_name=old_profiler
suffix=".log"
subdir_name=logs
#suffix=".timeline"
#subdir_name=timeline

model_name=yolov3_darknet53_270e_coco
output_filename=${model_name}_bs${batch_size}_${precision}_logiter${log_iter}.opt_${opt_level}
output_root=/root/paddlejob/workspace/work/zhangting/ModelPerf-AMP/PaddleDetection/${model_name}
mkdir -p ${output_root}/${subdir_name}


collect_gpu_status=False
if [ "${collect_gpu_status}" = "True" ]; then
  nvidia-smi -i ${CUDA_VISIBLE_DEVICES} --query-gpu=utilization.gpu,utilization.memory --format=csv -lms 100 > ${output_root}/${subdir_name}/gpu_usage_${output_filename}.txt 2>&1 &
  gpu_query_pid=$!
  echo "gpu_query_pid=${gpu_query_pid}"
fi

#nsys_args="nsys profile --stats true -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu --capture-range=cudaProfilerApi -x true --force-overwrite true -o ${output_filename}"

echo "======================================================"
echo "NVIDIA_TF32_OVERRIDE            : ${NVIDIA_TF32_OVERRIDE}"
echo "FLAGS_enable_eager_mode         : ${FLAGS_enable_eager_mode}"
echo "FLAGS_use_autotune              : ${FLAGS_use_autotune}"
echo "FLAGS_conv_workspace_size_limit : ${FLAGS_conv_workspace_size_limit}"
echo "use_shared_memory               : ${use_shared_memory}"
echo "num_workers                     : ${num_workers}"
echo "model_name                      : ${model_name}"
echo "output_filename                 : ${output_filename}"
echo "nsys_args                       : ${nsys_args}"
echo "======================================================"

${nsys_args} python -u tools/train.py \
    -c configs/yolov3/yolov3_darknet53_270e_coco.yml \
    -o LearningRate.base_lr=0.0001 \
    log_iter=${log_iter} ${amp_level_args} \
    use_gpu=True \
    save_dir=./test_tipc/output/yolov3_darknet53_270e_coco/benchmark_train/norm_train_gpus_0_autocast_fp32 \
    epoch=1 \
    pretrain_weights=https://paddledet.bj.bcebos.com/models/yolov3_darknet53_270e_coco.pdparams \
    worker_num=${num_workers} \
    data_format=${data_format} \
    TrainReader.batch_size=${batch_size} \
    TrainReader.use_shared_memory=${use_shared_memory} \
    filename=yolov3_darknet53_270e_coco ${amp_args} | tee ${output_root}/${subdir_name}/log_${output_filename}${suffix}.txt 2>&1

if [ "${collect_gpu_status}" = "True" ]; then
  kill ${gpu_query_pid}
fi


