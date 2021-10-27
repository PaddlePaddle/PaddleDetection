#!/usr/bin/env bash
set -xe
# Usage：CUDA_VISIBLE_DEVICES=0 bash benchmark/run_benchmark.sh ${run_mode} ${batch_size} ${fp_item} ${max_epoch} ${model_name}
python="python3.7"
# Parameter description
function _set_params(){
    run_mode=${1:-"sp"}            # sp|mp
    batch_size=${2:-"2"}
    fp_item=${3:-"fp32"}           # fp32|fp16
    max_epoch=${4:-"1"}
    model_name=${5:-"model_name"}
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}

    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=(${device})
    num_gpu_devices=${#arr[*]}
    log_file=${run_log_path}/${model_name}_${run_mode}_bs${batch_size}_${fp_item}_${num_gpu_devices}
}
function _train(){
    echo "Train on ${num_gpu_devices} GPUs"
    echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"

    # set runtime params
    set_optimizer_lr_sp=" "
    set_optimizer_lr_mp=" "
    # parse model_name
    case ${model_name} in
        faster_rcnn) model_yml="configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml"
            set_optimizer_lr_sp="LearningRate.base_lr=0.001" ;;
        fcos) model_yml="configs/fcos/fcos_r50_fpn_1x_coco.yml"
            set_optimizer_lr_sp="LearningRate.base_lr=0.001" ;;
        deformable_detr) model_yml="configs/deformable_detr/deformable_detr_r50_1x_coco.yml" ;;
        gfl) model_yml="configs/gfl/gfl_r50_fpn_1x_coco.yml"
            set_optimizer_lr_sp="LearningRate.base_lr=0.001" ;;
        hrnet) model_yml="configs/keypoint/hrnet/hrnet_w32_256x192.yml" ;;
        higherhrnet) model_yml="configs/keypoint/higherhrnet/higherhrnet_hrnet_w32_512.yml" ;;
        solov2) model_yml="configs/solov2/solov2_r50_fpn_1x_coco.yml" ;;
        jde) model_yml="configs/mot/jde/jde_darknet53_30e_1088x608.yml" ;;
        fairmot) model_yml="configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml" ;;
        *) echo "Undefined model_name"; exit 1;
    esac

    set_batch_size="TrainReader.batch_size=${batch_size}"
    set_max_epoch="epoch=${max_epoch}"
    set_log_iter="log_iter=1"
    if [ ${fp_item} = "fp16" ]; then
        set_fp_item="--fp16"
    else
        set_fp_item=" "
    fi

    case ${run_mode} in
        sp) train_cmd="${python} -u tools/train.py -c ${model_yml} ${set_fp_item} \
            -o ${set_batch_size} ${set_max_epoch} ${set_log_iter} ${set_optimizer_lr_sp}" ;;
        mp) train_cmd="${python} -m paddle.distributed.launch --log_dir=./mylog \
            --gpus=${CUDA_VISIBLE_DEVICES} tools/train.py -c ${model_yml} ${set_fp_item} \
            -o ${set_batch_size} ${set_max_epoch} ${set_log_iter} ${set_optimizer_lr_mp}"
            log_parse_file="mylog/workerlog.0" ;;
        *) echo "choose run_mode(sp or mp)"; exit 1;
    esac

    timeout 15m ${train_cmd} > ${log_file} 2>&1
    if [ $? -ne 0 ];then
        echo -e "${train_cmd}, FAIL"
        export job_fail_flag=1
    else
        echo -e "${train_cmd}, SUCCESS"
        export job_fail_flag=0
    fi
    kill -9 `ps -ef|grep 'python'|awk '{print $2}'`

    if [ $run_mode = "mp" -a -d mylog ]; then
        rm ${log_file}
        cp mylog/workerlog.0 ${log_file}
    fi
}

_set_params $@
_train
