#!/bin/bash
source test_tipc/utils_func.sh

FILENAME=$1

# parser model_name
dataline=$(cat ${FILENAME})
IFS=$'\n'
lines=(${dataline})
model_name=$(func_parser_value "${lines[1]}")
echo "ppdet serving: ${model_name}"
python=$(func_parser_value "${lines[2]}")
filename_key=$(func_parser_key "${lines[3]}")
filename_value=$(func_parser_value "${lines[3]}")

# export params
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
export_serving_model_key=$(func_parser_key "${lines[14]}")
export_serving_model_value=$(func_parser_value "${lines[14]}")
# parser serving
start_serving=$(func_parser_value "${lines[16]}")
port_key=$(func_parser_key "${lines[17]}")
port_value=$(func_parser_value "${lines[17]}")
gpu_id_key=$(func_parser_key "${lines[18]}")
gpu_id_value=$(func_parser_value "${lines[18]}")

LOG_PATH="./test_tipc/output"
mkdir -p ${LOG_PATH}
status_log="${LOG_PATH}/results_serving.log"

function func_serving(){
    IFS='|'
    if [ ${gpu_id_key} = "null" ]; then
        start_serving_command="nohup ${python} ${start_serving} ${port_key} ${port_value} > serving.log 2>&1 &"
    else
        start_serving_command="nohup ${python} ${start_serving} ${port_key} ${port_value} ${gpu_id_key} ${gpu_id_value} > serving.log 2>&1 &"
    fi
    echo $start_serving_command
    eval $start_serving_command
    last_status=${PIPESTATUS[0]}
    status_check $last_status "${start_serving_command}" "${status_log}"
}
cd output_inference/${model_name}
echo $PWD
func_serving
test_command="${python} ../../deploy/serving/test_client.py ../../deploy/serving/label_list.txt ../../demo/000000014439.jpg"
echo $test_command
eval $test_command
last_status=${PIPESTATUS[0]}
status_check $last_status"${test_command}" "${status_log}"

