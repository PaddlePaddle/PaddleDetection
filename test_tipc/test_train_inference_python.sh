#!/bin/bash
source test_tipc/utils_func.sh

FILENAME=$1
# MODE be one of ['lite_train_lite_infer' 'lite_train_whole_infer'
#                 'whole_train_whole_infer', 'whole_infer', 'klquant_whole_infer']
MODE=$2

# parse params
dataline=$(cat ${FILENAME})
IFS=$'\n'
lines=(${dataline})

# The training params
model_name=$(func_parser_value "${lines[1]}")
echo "ppdet python_infer: ${model_name}"
python=$(func_parser_value "${lines[2]}")
gpu_list=$(func_parser_value "${lines[3]}")
train_use_gpu_key=$(func_parser_key "${lines[4]}")
train_use_gpu_value=$(func_parser_value "${lines[4]}")
autocast_list=$(func_parser_value "${lines[5]}")
autocast_key=$(func_parser_key "${lines[5]}")
epoch_key=$(func_parser_key "${lines[6]}")
epoch_num=$(func_parser_params "${lines[6]}")
save_model_key=$(func_parser_key "${lines[7]}")
train_batch_key=$(func_parser_key "${lines[8]}")
train_batch_value=$(func_parser_params "${lines[8]}")
pretrain_model_key=$(func_parser_key "${lines[9]}")
pretrain_model_value=$(func_parser_value "${lines[9]}")
train_model_name=$(func_parser_value "${lines[10]}")
train_infer_img_dir=$(func_parser_value "${lines[11]}")
train_param_key1=$(func_parser_key "${lines[12]}")
train_param_value1=$(func_parser_value "${lines[12]}")

trainer_list=$(func_parser_value "${lines[14]}")
norm_key=$(func_parser_key "${lines[15]}")
norm_trainer=$(func_parser_value "${lines[15]}")
pact_key=$(func_parser_key "${lines[16]}")
pact_trainer=$(func_parser_value "${lines[16]}")
fpgm_key=$(func_parser_key "${lines[17]}")
fpgm_trainer=$(func_parser_value "${lines[17]}")
distill_key=$(func_parser_key "${lines[18]}")
distill_trainer=$(func_parser_value "${lines[18]}")
trainer_key1=$(func_parser_key "${lines[19]}")
trainer_value1=$(func_parser_value "${lines[19]}")
trainer_key2=$(func_parser_key "${lines[20]}")
trainer_value2=$(func_parser_value "${lines[20]}")

# eval params
eval_py=$(func_parser_value "${lines[23]}")
eval_key1=$(func_parser_key "${lines[24]}")
eval_value1=$(func_parser_value "${lines[24]}")

# export params
save_export_key=$(func_parser_key "${lines[27]}")
save_export_value=$(func_parser_value "${lines[27]}")
export_weight_key=$(func_parser_key "${lines[28]}")
export_weight_value=$(func_parser_value "${lines[28]}")
norm_export=$(func_parser_value "${lines[29]}")
pact_export=$(func_parser_value "${lines[30]}")
fpgm_export=$(func_parser_value "${lines[31]}")
distill_export=$(func_parser_value "${lines[32]}")
export_key1=$(func_parser_key "${lines[33]}")
export_value1=$(func_parser_value "${lines[33]}")
export_key2=$(func_parser_key "${lines[34]}")
export_value2=$(func_parser_value "${lines[34]}")
kl_quant_export=$(func_parser_value "${lines[35]}")

# parser inference model
infer_mode_list=$(func_parser_value "${lines[37]}")
infer_is_quant_list=$(func_parser_value "${lines[38]}")
# parser inference
inference_py=$(func_parser_value "${lines[39]}")
use_gpu_key=$(func_parser_key "${lines[40]}")
use_gpu_list=$(func_parser_value "${lines[40]}")
use_mkldnn_key=$(func_parser_key "${lines[41]}")
use_mkldnn_list=$(func_parser_value "${lines[41]}")
cpu_threads_key=$(func_parser_key "${lines[42]}")
cpu_threads_list=$(func_parser_value "${lines[42]}")
batch_size_key=$(func_parser_key "${lines[43]}")
batch_size_list=$(func_parser_value "${lines[43]}")
use_trt_key=$(func_parser_key "${lines[44]}")
use_trt_list=$(func_parser_value "${lines[44]}")
precision_key=$(func_parser_key "${lines[45]}")
precision_list=$(func_parser_value "${lines[45]}")
infer_model_key=$(func_parser_key "${lines[46]}")
image_dir_key=$(func_parser_key "${lines[47]}")
infer_img_dir=$(func_parser_value "${lines[47]}")
save_log_key=$(func_parser_key "${lines[48]}")
benchmark_key=$(func_parser_key "${lines[49]}")
benchmark_value=$(func_parser_value "${lines[49]}")
infer_key1=$(func_parser_key "${lines[50]}")
infer_value1=$(func_parser_value "${lines[50]}")

LOG_PATH="./test_tipc/output"
mkdir -p ${LOG_PATH}
status_log="${LOG_PATH}/results_python.log"


function func_inference(){
    IFS='|'
    _python=$1
    _script=$2
    _model_dir=$3
    _log_path=$4
    _img_dir=$5
    _flag_quant=$6
    # inference
    for use_gpu in ${use_gpu_list[*]}; do
        if [ ${use_gpu} = "False" ] || [ ${use_gpu} = "cpu" ]; then
            for use_mkldnn in ${use_mkldnn_list[*]}; do
                if [ ${use_mkldnn} = "False" ] && [ ${_flag_quant} = "True" ]; then
                    continue
                fi
                for threads in ${cpu_threads_list[*]}; do
                    for batch_size in ${batch_size_list[*]}; do
                        _save_log_path="${_log_path}/python_infer_cpu_usemkldnn_${use_mkldnn}_threads_${threads}_precision_fluid_batchsize_${batch_size}.log"
                        set_infer_data=$(func_set_params "${image_dir_key}" "${_img_dir}")
                        set_benchmark=$(func_set_params "${benchmark_key}" "${benchmark_value}")
                        set_batchsize=$(func_set_params "${batch_size_key}" "${batch_size}")
                        set_cpu_threads=$(func_set_params "${cpu_threads_key}" "${threads}")
                        set_model_dir=$(func_set_params "${infer_model_key}" "${_model_dir}")
                        set_infer_params1=$(func_set_params "${infer_key1}" "${infer_value1}")
                        command="${_python} ${_script} ${use_gpu_key}=${use_gpu} ${use_mkldnn_key}=${use_mkldnn} ${set_cpu_threads} ${set_model_dir} ${set_batchsize} ${set_infer_data} ${set_benchmark} ${set_infer_params1} > ${_save_log_path} 2>&1 "
                        eval $command
                        last_status=${PIPESTATUS[0]}
                        eval "cat ${_save_log_path}"
                        status_check $last_status "${command}" "${status_log}"
                    done
                done
            done
        elif [ ${use_gpu} = "True" ] || [ ${use_gpu} = "gpu" ]; then
            for precision in ${precision_list[*]}; do
                if [[ ${precision} != "fluid" ]]; then
                    if [[ ${_flag_quant} = "False" ]] && [[ ${precision} = "trt_int8" ]]; then
                        continue
                    fi
                    if [[ ${_flag_quant} = "True" ]] && [[ ${precision} != "trt_int8" ]]; then
                        continue
                    fi
                fi
                for batch_size in ${batch_size_list[*]}; do
                    _save_log_path="${_log_path}/python_infer_gpu_precision_${precision}_batchsize_${batch_size}.log"
                    set_infer_data=$(func_set_params "${image_dir_key}" "${_img_dir}")
                    set_benchmark=$(func_set_params "${benchmark_key}" "${benchmark_value}")
                    set_batchsize=$(func_set_params "${batch_size_key}" "${batch_size}")
                    set_precision=$(func_set_params "${precision_key}" "${precision}")
                    set_model_dir=$(func_set_params "${infer_model_key}" "${_model_dir}")
                    set_infer_params1=$(func_set_params "${infer_key1}" "${infer_value1}")
                    command="${_python} ${_script} ${use_gpu_key}=${use_gpu} ${set_precision} ${set_model_dir} ${set_batchsize} ${set_infer_data} ${set_benchmark} ${set_infer_params1} > ${_save_log_path} 2>&1 "
                    eval $command
                    last_status=${PIPESTATUS[0]}
                    eval "cat ${_save_log_path}"
                    status_check $last_status "${command}" "${status_log}"
                done
            done
        else
            echo "Does not support hardware other than CPU and GPU Currently!"
        fi
    done
}

if [ ${MODE} = "whole_infer" ] || [ ${MODE} = "klquant_whole_infer" ]; then
    # set CUDA_VISIBLE_DEVICES
    GPUID=$3
    if [ ${#GPUID} -le 0 ];then
        env=" "
    else
        env="export CUDA_VISIBLE_DEVICES=${GPUID}"
    fi
    eval $env

    Count=0
    IFS="|"
    infer_quant_flag=(${infer_is_quant_list})
    for infer_mode in ${infer_mode_list[*]}; do
        if [ ${infer_mode} = "null" ]; then
            continue
        fi
        if [ ${MODE} = "klquant_whole_infer" ] && [ ${infer_mode} != "kl_quant" ]; then
            continue
        fi
        if [ ${MODE} = "whole_infer" ] && [ ${infer_mode} = "kl_quant" ]; then
            continue
        fi
        # run export
        case ${infer_mode} in
            norm) run_export=${norm_export} ;;
            pact) run_export=${pact_export} ;;
            fpgm) run_export=${fpgm_export} ;;
            distill) run_export=${distill_export} ;;
            kl_quant) run_export=${kl_quant_export} ;;
            *) echo "Undefined infer_mode!"; exit 1;
        esac
        set_export_weight=$(func_set_params "${export_weight_key}" "${export_weight_value}")
        set_save_export_dir=$(func_set_params "${save_export_key}" "${save_export_value}")
        set_filename=$(func_set_params "filename" "${model_name}")
        export_cmd="${python} ${run_export} ${set_export_weight} ${set_filename} ${set_save_export_dir} "
        echo  $export_cmd
        eval $export_cmd
        status_check $? "${export_cmd}" "${status_log}"

        #run inference
        save_export_model_dir="${save_export_value}/${model_name}"
        is_quant=${infer_quant_flag[Count]}
        func_inference "${python}" "${inference_py}" "${save_export_model_dir}" "${LOG_PATH}" "${infer_img_dir}" ${is_quant}
        Count=$((${Count} + 1))
    done
else
    IFS="|"
    Count=0
    for gpu in ${gpu_list[*]}; do
        use_gpu=${train_use_gpu_value}
        Count=$((${Count} + 1))
        ips=""
        if [ ${gpu} = "-1" ];then
            env=""
            use_gpu=False
        elif [ ${#gpu} -le 1 ];then
            env="export CUDA_VISIBLE_DEVICES=${gpu}"
            eval ${env}
        elif [ ${#gpu} -le 15 ];then
            IFS=","
            array=(${gpu})
            env="export CUDA_VISIBLE_DEVICES=${array[0]}"
            IFS="|"
        else
            IFS=";"
            array=(${gpu})
            ips=${array[0]}
            gpu=${array[1]}
            IFS="|"
            env=" "
        fi
        for autocast in ${autocast_list[*]}; do
            for trainer in ${trainer_list[*]}; do
                flag_quant=False
                if [ ${trainer} = "${norm_key}" ]; then
                    run_train=${norm_trainer}
                    run_export=${norm_export}
                elif [ ${trainer} = "${pact_key}" ]; then
                    run_train=${pact_trainer}
                    run_export=${pact_export}
                    flag_quant=True
                elif [ ${trainer} = "${fpgm_key}" ]; then
                    run_train=${fpgm_trainer}
                    run_export=${fpgm_export}
                elif [ ${trainer} = "${distill_key}" ]; then
                    run_train=${distill_trainer}
                    run_export=${distill_export}
                elif [ ${trainer} = "${trainer_key1}" ]; then
                    run_train=${trainer_value1}
                    run_export=${export_value1}
                elif [ ${trainer} = "${trainer_key2}" ]; then
                    run_train=${trainer_value2}
                    run_export=${export_value2}
                else
                    continue
                fi

                if [ ${run_train} = "null" ]; then
                    continue
                fi

                if [ ${autocast} = "amp" ]; then
                    set_autocast="--amp"
                else
                    set_autocast=" "
                fi
                set_epoch=$(func_set_params "${epoch_key}" "${epoch_num}")
                set_pretrain=$(func_set_params "${pretrain_model_key}" "${pretrain_model_value}")
                set_batchsize=$(func_set_params "${train_batch_key}" "${train_batch_value}")
                set_filename=$(func_set_params "filename" "${model_name}")
                set_use_gpu=$(func_set_params "${train_use_gpu_key}" "${use_gpu}")
                set_train_params1=$(func_set_params "${train_param_key1}" "${train_param_value1}")
                save_log="${LOG_PATH}/${trainer}_gpus_${gpu}_autocast_${autocast}"

                set_save_model=$(func_set_params "${save_model_key}" "${save_log}")
                if [ ${#gpu} -le 2 ];then  # train with cpu or single gpu
                    cmd="${python} ${run_train} LearningRate.base_lr=0.0001 log_iter=1 ${set_use_gpu} ${set_save_model} ${set_epoch} ${set_pretrain} ${set_batchsize} ${set_filename} ${set_train_params1} ${set_autocast}"
                elif [ ${#ips} -le 26 ];then  # train with multi-gpu
                    cmd="${python} -m paddle.distributed.launch --gpus=${gpu} ${run_train} log_iter=1 ${set_use_gpu} ${set_save_model} ${set_epoch} ${set_pretrain} ${set_batchsize} ${set_filename} ${set_train_params1} ${set_autocast}"
                else     # train with multi-machine
                    cmd="${python} -m paddle.distributed.launch --ips=${ips} --gpus=${gpu} ${set_use_gpu} ${run_train} log_iter=1 ${set_save_model} ${set_epoch} ${set_pretrain} ${set_batchsize} ${set_filename} ${set_train_params1} ${set_autocast}"
                fi
                # run train
                eval $cmd
                status_check $? "${cmd}" "${status_log}"

                set_eval_trained_weight=$(func_set_params "${export_weight_key}" "${save_log}/${model_name}/${train_model_name}")
                # run eval
                if [ ${eval_py} != "null" ]; then
                    set_eval_params1=$(func_set_params "${eval_key1}" "${eval_value1}")
                    eval_cmd="${python} ${eval_py} ${set_eval_trained_weight} ${set_use_gpu} ${set_eval_params1}"
                    eval $eval_cmd
                    status_check $? "${eval_cmd}" "${status_log}"
                fi
                # run export model
                if [ ${run_export} != "null" ]; then
                    # run export model
                    set_export_weight=$(func_set_params "${export_weight_key}" "${save_log}/${model_name}/${train_model_name}")
                    set_save_export_dir=$(func_set_params "${save_export_key}" "${save_log}")
                    export_cmd="${python} ${run_export} ${set_export_weight} ${set_filename} ${set_save_export_dir} "
                    eval $export_cmd
                    status_check $? "${export_cmd}" "${status_log}"

                    #run inference
                    save_export_model_dir="${save_export_value}/${model_name}"
                    eval $env
                    func_inference "${python}" "${inference_py}" "${save_export_model_dir}" "${LOG_PATH}" "${train_infer_img_dir}" "${flag_quant}"

                    eval "unset CUDA_VISIBLE_DEVICES"
                fi
            done  # done with:    for trainer in ${trainer_list[*]}; do
        done      # done with:    for autocast in ${autocast_list[*]}; do
    done          # done with:    for gpu in ${gpu_list[*]}; do
fi  # end if [ ${MODE} = "infer" ]; then
