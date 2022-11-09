#!/bin/bash

# PR: https://github.com/PaddlePaddle/PaddleDetection/pull/6483
batch_size=${1:-8}
precision=${2:-"fp32"} # tf32, fp16_o1, fp16_o2
opt_level=${3:-4}
echo "batch_size=${batch_size}, precision=${precision}, opt_level=${opt_level}"

export CUDA_VISIBLE_DEVICES="3"

WORK_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/" && pwd )"
export PYTHONPATH=${WORK_ROOT}:${PYTHONPATH}
export GLOG_vmodule=switch_autotune=2
#export GLOG_vmodule=eager_final_state_op_function=6,api=6,dygraph_api=6,apu_custom_impl=6,dygraph_functions=6,amp_utils=6,data_transform=6,eager_amp_auto_cast=6
#export GLOG_v=6

log_iter=10

# opt 1
if [ ${opt_level} -ge 1 ]; then
  use_shared_memory=True
else
  use_shared_memory=False
fi

# opt 2
if [ ${opt_level} -ge 2 ]; then
  if [ ${batch_size} -eq 8 ]; then
    num_workers=16
  else
    num_workers=2
  fi
else
  num_workers=2
fi

# opt 3
if [ ${opt_level} -ge 3 ]; then
  export FLAGS_use_autotune=1
  #export FLAGS_cudnn_exhaustive_search=1
else
  export FLAGS_use_autotune=0
fi

# opt 4
if [ ${opt_level} -ge 4 ]; then
  export FLAGS_conv_workspace_size_limit=4000 #MB
fi

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

#suffix=".profile-native_old"
subdir_name=logs_dev20220805
model_name=fcos_r50_fpn_1x_coco
output_filename=level_group_nhwc_${model_name}_bs${batch_size}_${precision}_logiter${log_iter}.opt_${opt_level}
output_root=/root/paddlejob/workspace/work/liuyiqun/ModelPerf-AMP/PaddleDetection/${model_name}
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

${nsys_args} python -u ${WORK_ROOT}/tools/train.py \
    -c configs/fcos/fcos_r50_fpn_1x_coco.yml \
    -o LearningRate.base_lr=0.0001 \
    log_iter=${log_iter} ${amp_level_args} \
    use_gpu=True \
    epoch=1 \
    save_dir=./test_tipc/output/fcos_r50_fpn_1x_coco/benchmark_train/norm_train_gpus_0_autocast_fp32 \
    pretrain_weights=https://paddledet.bj.bcebos.com/models/fcos_r50_fpn_1x_coco.pdparams \
    worker_num=${num_workers} \
    TrainReader.batch_size=${batch_size} \
    TrainReader.use_shared_memory=${use_shared_memory} \
    filename=fcos_r50_fpn_1x_coco ${amp_args} | tee ${output_root}/${subdir_name}/log_${output_filename}${suffix}.txt 2>&1

if [ "${collect_gpu_status}" = "True" ]; then
  kill ${gpu_query_pid}
fi

