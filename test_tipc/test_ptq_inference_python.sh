#!/bin/bash
source test_tipc/utils_func.sh

FILENAME=$1
MODE="whole_infer"

# parser model_name
dataline=$(cat ${FILENAME})
IFS=$'\n'
lines=(${dataline})
model_name=$(func_parser_value "${lines[1]}")
echo "ppdet ptq: ${model_name}"
python=$(func_parser_value "${lines[2]}")
filename_key=$(func_parser_key "${lines[3]}")

# parser export params
save_export_key=$(func_parser_key "${lines[5]}")
save_export_value=$(func_parser_value "${lines[5]}")
export_weight_key=$(func_parser_key "${lines[6]}")
export_weight_value=$(func_parser_value "${lines[6]}")
kl_quant_export=$(func_parser_value "${lines[7]}")
export_param1_key=$(func_parser_key "${lines[8]}")
export_param1_value=$(func_parser_value "${lines[8]}")

# parser infer params
inference_py=$(func_parser_value "${lines[10]}")
device_key=$(func_parser_key "${lines[11]}")
device_list=$(func_parser_value "${lines[11]}")
use_mkldnn_key=$(func_parser_key "${lines[12]}")
use_mkldnn_list=$(func_parser_value "${lines[12]}")
cpu_threads_key=$(func_parser_key "${lines[13]}")
cpu_threads_list=$(func_parser_value "${lines[13]}")
batch_size_key=$(func_parser_key "${lines[14]}")
batch_size_list=$(func_parser_value "${lines[14]}")
run_mode_key=$(func_parser_key "${lines[15]}")
run_mode_list=$(func_parser_value "${lines[15]}")
model_dir_key=$(func_parser_key "${lines[16]}")
image_dir_key=$(func_parser_key "${lines[17]}")
image_dir_value=$(func_parser_value "${lines[17]}")
run_benchmark_key=$(func_parser_key "${lines[18]}")
run_benchmark_value=$(func_parser_value "${lines[18]}")
infer_param1_key=$(func_parser_key "${lines[19]}")
infer_param1_value=$(func_parser_value "${lines[19]}")


LOG_PATH="./test_tipc/output/${model_name}/${MODE}"
mkdir -p ${LOG_PATH}
status_log="${LOG_PATH}/results_ptq_python.log"

function func_ptq_inference(){
    IFS='|'
    _python=$1
    _log_path=$2
    _script=$3
    _set_model_dir=$4

    set_image_dir=$(func_set_params "${image_dir_key}" "${image_dir_value}")
    set_run_benchmark=$(func_set_params "${run_benchmark_key}" "${run_benchmark_value}")
    set_infer_param1=$(func_set_params "${infer_param1_key}" "${infer_param1_value}")
    # inference
    for device in ${device_list[*]}; do
        set_device=$(func_set_params "${device_key}" "${device}")
        if [ ${device} = "cpu" ]; then
            for use_mkldnn in ${use_mkldnn_list[*]}; do
                set_use_mkldnn=$(func_set_params "${use_mkldnn_key}" "${use_mkldnn}")
                for threads in ${cpu_threads_list[*]}; do
                    set_cpu_threads=$(func_set_params "${cpu_threads_key}" "${threads}")
                    for batch_size in ${batch_size_list[*]}; do
                        _save_log_path="${_log_path}/python_infer_cpu_usemkldnn_${use_mkldnn}_threads_${threads}_mode_paddle_batchsize_${batch_size}.log"
                        set_batchsize=$(func_set_params "${batch_size_key}" "${batch_size}")
                        command="${_python} ${_script} ${set_device} ${set_use_mkldnn} ${set_cpu_threads} ${_set_model_dir} ${set_batchsize} ${set_image_dir} ${set_run_benchmark} ${set_infer_param1} > ${_save_log_path} 2>&1 "
                        eval $command
                        last_status=${PIPESTATUS[0]}
                        eval "cat ${_save_log_path}"
                        status_check $last_status "${command}" "${status_log}" "${model_name}" "${_save_log_path}"
                    done
                done
            done
        elif [ ${device} = "gpu" ]; then
            for run_mode in ${run_mode_list[*]}; do
                if [[ ${run_mode} = "paddle" ]] || [[ ${run_mode} = "trt_int8" ]]; then
                    for batch_size in ${batch_size_list[*]}; do
                        _save_log_path="${_log_path}/python_infer_gpu_mode_${run_mode}_batchsize_${batch_size}.log"
                        set_batchsize=$(func_set_params "${batch_size_key}" "${batch_size}")
                        set_run_mode=$(func_set_params "${run_mode_key}" "${run_mode}")
                        command="${_python} ${_script} ${set_device} ${set_run_mode} ${_set_model_dir} ${set_batchsize} ${set_image_dir} ${set_run_benchmark} ${set_infer_param1} > ${_save_log_path} 2>&1 "
                        eval $command
                        last_status=${PIPESTATUS[0]}
                        eval "cat ${_save_log_path}"
                        status_check $last_status "${command}" "${status_log}" "${model_name}" "${_save_log_path}"
                    done
                fi
            done
        else
            echo "Does not support hardware other than CPU and GPU Currently!"
        fi
    done
}

IFS="|"
# run ptq
set_export_weight=$(func_set_params "${export_weight_key}" "${export_weight_value}")
set_save_export_dir=$(func_set_params "${save_export_key}" "${save_export_value}")
set_filename=$(func_set_params "${filename_key}" "${model_name}")
export_log_path="${LOG_PATH}/export.log"
ptq_cmd="${python} ${kl_quant_export} ${set_export_weight} ${set_filename} ${set_save_export_dir}"
echo  $ptq_cmd
eval "${ptq_cmd} > ${export_log_path} 2>&1"
status_export=$?
cat ${export_log_path}
status_check $status_export "${ptq_cmd}" "${status_log}" "${model_name}" "${export_log_path}"

#run inference
set_export_model_dir=$(func_set_params "${model_dir_key}" "${save_export_value}/${model_name}")
func_ptq_inference "${python}" "${LOG_PATH}" "${inference_py}" "${set_export_model_dir}"
