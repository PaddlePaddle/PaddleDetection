#!/bin/bash
source test_tipc/utils_func.sh

FILENAME=$1

# parser model_name
dataline=$(cat ${FILENAME})
IFS=$'\n'
lines=(${dataline})
model_name=$(func_parser_value "${lines[1]}")
echo "ppdet onnx_infer: ${model_name}"
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

# parser paddle2onnx
padlle2onnx_cmd=$(func_parser_value "${lines[15]}")
infer_model_dir_key=$(func_parser_key "${lines[16]}")
infer_model_dir_value=$(func_parser_value "${lines[16]}")
model_filename_key=$(func_parser_key "${lines[17]}")
model_filename_value=$(func_parser_value "${lines[17]}")
params_filename_key=$(func_parser_key "${lines[18]}")
params_filename_value=$(func_parser_value "${lines[18]}")
save_file_key=$(func_parser_key "${lines[19]}")
save_file_value=$(func_parser_value "${lines[19]}")
opset_version_key=$(func_parser_key "${lines[20]}")
opset_version_value=$(func_parser_value "${lines[20]}")

# parser onnx inference 
inference_py=$(func_parser_value "${lines[22]}")
model_file_key=$(func_parser_key "${lines[23]}")
model_file_value=$(func_parser_value "${lines[23]}")
img_fold_key=$(func_parser_key "${lines[24]}")
img_fold_value=$(func_parser_value "${lines[24]}")
results_fold_key=$(func_parser_key "${lines[25]}")
results_fold_value=$(func_parser_value "${lines[25]}")
onnx_infer_mode_list=$(func_parser_value "${lines[26]}")

LOG_PATH="./test_tipc/output"
mkdir -p ${LOG_PATH}
status_log="${LOG_PATH}/results_paddle2onnx.log"

function func_paddle2onnx(){
    IFS='|'
    _script=$1

    # paddle2onnx
    echo "################### run onnx export ###################"
    _save_log_path="${LOG_PATH}/paddle2onnx_infer_cpu.log"
    set_dirname=$(func_set_params "${infer_model_dir_key}" "${infer_model_dir_value}")
    set_model_filename=$(func_set_params "${model_filename_key}" "${model_filename_value}")
    set_params_filename=$(func_set_params "${params_filename_key}" "${params_filename_value}")
    set_save_model=$(func_set_params "${save_file_key}" "${save_file_value}")
    set_opset_version=$(func_set_params "${opset_version_key}" "${opset_version_value}")
    trans_model_cmd="${padlle2onnx_cmd} ${set_dirname} ${set_model_filename} ${set_params_filename} ${set_save_model} ${set_opset_version}"
    eval $trans_model_cmd
    last_status=${PIPESTATUS[0]}
    status_check $last_status "${trans_model_cmd}" "${status_log}"
    # python inference
    echo "################### run infer ###################"
    cd ./deploy/third_engine/demo_onnxruntime/
    model_file=$(func_set_params "${model_file_key}" "${model_file_value}")
    img_fold=$(func_set_params "${img_fold_key}" "${img_fold_value}")
    results_fold=$(func_set_params "${results_fold_key}" "${results_fold_value}")
    infer_model_cmd="${python} ${inference_py} ${model_file} ${img_fold} ${results_fold}"
    eval $infer_model_cmd
    last_status=${PIPESTATUS[0]}
    status_check $last_status "${infer_model_cmd}" "${status_log}"
}

export Count=0
IFS="|"
echo "################### run paddle export ###################"
for infer_mode in ${onnx_infer_mode_list[*]}; do

    # run export
    case ${infer_mode} in
        norm) run_export=${norm_export} ;;
        quant) run_export=${pact_export} ;;
        fpgm) run_export=${fpgm_export} ;;
        distill) run_export=${distill_export} ;;
        kl_quant) run_export=${kl_quant_export} ;;
        *) echo "Undefined infer_mode!"; exit 1;
    esac
    if [ ${run_export} = "null" ]; then
        continue
    fi
    set_export_weight=$(func_set_params "${export_weight_key}" "${export_weight_value}")
    set_save_export_dir=$(func_set_params "${save_export_key}" "${save_export_value}")
    set_filename=$(func_set_params "${filename_key}" "${model_name}")
    export_cmd="${python} ${run_export} ${set_export_weight} ${set_filename} ${set_save_export_dir} "
    echo  $export_cmd
    eval $export_cmd
    status_export=$?
    status_check $status_export "${export_cmd}" "${status_log}"
done
func_paddle2onnx 