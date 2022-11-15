#!/bin/bash
source test_tipc/utils_func.sh

FILENAME=$1
MODE="serving_infer"

# parser model_name
dataline=$(cat ${FILENAME})
IFS=$'\n'
lines=(${dataline})
model_name=$(func_parser_value "${lines[1]}")
echo "ppdet serving_cpp_infer: ${model_name}"
python=$(func_parser_value "${lines[2]}")
filename_key=$(func_parser_key "${lines[3]}")
filename_value=$(func_parser_value "${lines[3]}")

# parser export params
save_export_key=$(func_parser_key "${lines[5]}")
save_export_value=$(func_parser_value "${lines[5]}")
export_weight_key=$(func_parser_key "${lines[6]}")
export_weight_value=$(func_parser_value "${lines[6]}")
norm_export=$(func_parser_value "${lines[7]}")
pact_export=$(func_parser_value "${lines[8]}")
fpgm_export=$(func_parser_value "${lines[9]}")
distill_export=$(func_parser_value "${lines[10]}")
export_key1=$(func_parser_key "${lines[11]}")
export_value1=$(func_parser_value "${lines[11]}")
export_key2=$(func_parser_key "${lines[12]}")
export_value2=$(func_parser_value "${lines[12]}")
kl_quant_export=$(func_parser_value "${lines[13]}")

# parser serving params
infer_mode_list=$(func_parser_value "${lines[15]}")
infer_is_quant_list=$(func_parser_value "${lines[16]}")

model_key=$(func_parser_key "${lines[17]}")
op_key=$(func_parser_key "${lines[18]}")
op_value=$(func_parser_value "${lines[18]}")
port_key=$(func_parser_key "${lines[19]}")
port_value=$(func_parser_value "${lines[19]}")
gpu_ids_key=$(func_parser_key "${lines[20]}")
gpu_ids_value=$(func_parser_value "${lines[20]}")
web_service_key1=$(func_parser_key "${lines[21]}")
web_service_value1=$(func_parser_value "${lines[21]}")
http_client_py=$(func_parser_value "${lines[22]}")
serving_client_key=$(func_parser_key "${lines[23]}")
infer_image_key=$(func_parser_key "${lines[24]}")
infer_image_value=$(func_parser_value "${lines[24]}")
http_client_key1=$(func_parser_key "${lines[25]}")
http_client_value1=$(func_parser_value "${lines[25]}")

LOG_PATH="./test_tipc/output/${model_name}/${MODE}"
mkdir -p ${LOG_PATH}
status_log="${LOG_PATH}/results_serving_cpp.log"

function func_serving_inference(){
    IFS='|'
    _python=$1
    _log_path=$2
    _set_server_model_dir=$3
    _set_client_model_dir=$4
    _set_image_file=$5

    set_op=$(func_set_params "${op_key}" "${op_value}")
    set_port=$(func_set_params "${port_key}" "${port_value}")
    set_web_service_params1=$(func_set_params "${web_service_key1}" "${web_service_value1}")
    set_http_client_params1=$(func_set_params "${http_client_key1}" "${http_client_value1}")
    # inference
    for gpu_ids in ${gpu_ids_value[*]}; do
        if [ ${gpu_ids} = "null" ];then
            server_log_path="${_log_path}/cpp_server_cpu.log"
            client_log_path="${_log_path}/cpp_client_cpu.log"
        else
            server_log_path="${_log_path}/cpp_server_gpu.log"
            client_log_path="${_log_path}/cpp_client_gpu.log"
        fi
        set_gpu_ids=$(func_set_params "${gpu_ids_key}" "${gpu_ids}")
        # run web service
        web_service_cmd="${_python} -m paddle_serving_server.serve ${_set_server_model_dir} ${set_op} ${set_port} ${set_gpu_ids} ${set_web_service_params1} > ${server_log_path} 2>&1 &"
        eval $web_service_cmd
        last_status=${PIPESTATUS[0]}
        cat ${server_log_path}
        status_check $last_status "${web_service_cmd}" "${status_log}" "${model_name}" "${server_log_path}"
        sleep 5s
        # run http client
        http_client_cmd="${_python} ${http_client_py} ${_set_client_model_dir} ${_set_image_file} ${set_http_client_params1} > ${client_log_path} 2>&1"
        eval $http_client_cmd
        last_status=${PIPESTATUS[0]}
        cat ${client_log_path}
        status_check $last_status "${http_client_cmd}" "${status_log}" "${model_name}" "${client_log_path}"
        ps ux | grep -i ${port_value} | awk '{print $2}' | xargs kill -s 9
        sleep 2s
    done
}

# run serving infer
Count=0
IFS="|"
infer_quant_flag=(${infer_is_quant_list})
for infer_mode in ${infer_mode_list[*]}; do
    if [ ${infer_mode} != "null" ]; then
        # run export
        case ${infer_mode} in
            norm) run_export=${norm_export} ;;
            quant) run_export=${pact_export} ;;
            fpgm) run_export=${fpgm_export} ;;
            distill) run_export=${distill_export} ;;
            kl_quant) run_export=${kl_quant_export} ;;
            *) echo "Undefined infer_mode!"; exit 1;
        esac
        set_export_weight=$(func_set_params "${export_weight_key}" "${export_weight_value}")
        set_save_export_dir=$(func_set_params "${save_export_key}" "${save_export_value}")
        set_filename=$(func_set_params "${filename_key}" "${model_name}")
        export_log_path="${LOG_PATH}/export.log"
        export_cmd="${python} ${run_export} ${set_export_weight} ${set_filename} ${set_save_export_dir} "
        echo  $export_cmd
        eval "${export_cmd} > ${export_log_path} 2>&1"
        status_export=$?
        cat ${export_log_path}
        status_check $status_export "${export_cmd}" "${status_log}" "${model_name}" "${export_log_path}"
    fi

    #run inference
    set_server_model_dir=$(func_set_params "${model_key}" "${save_export_value}/${model_name}/serving_server")
    set_client_model_dir=$(func_set_params "${serving_client_key}" "${save_export_value}/${model_name}/serving_client")
    set_infer_image_file=$(func_set_params "${infer_image_key}" "${infer_image_value}")
    is_quant=${infer_quant_flag[Count]}
    func_serving_inference "${python}" "${LOG_PATH}" "${set_server_model_dir}" "${set_client_model_dir}" ${set_infer_image_file}
    Count=$(($Count + 1))
done
eval "unset CUDA_VISIBLE_DEVICES"
