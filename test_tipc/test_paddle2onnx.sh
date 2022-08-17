#!/bin/bash
source test_tipc/utils_func.sh

FILENAME=$1
MODE="paddle2onnx_infer"

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
export_param_key=$(func_parser_key "${lines[12]}")
export_param_value=$(func_parser_value "${lines[12]}")
kl_quant_export=$(func_parser_value "${lines[13]}")

# parser paddle2onnx params
infer_mode_list=$(func_parser_value "${lines[15]}")
infer_is_quant_list=$(func_parser_value "${lines[16]}")

padlle2onnx_cmd=$(func_parser_value "${lines[17]}")
model_dir_key=$(func_parser_key "${lines[18]}")
model_filename_key=$(func_parser_key "${lines[19]}")
model_filename_value=$(func_parser_value "${lines[19]}")
params_filename_key=$(func_parser_key "${lines[20]}")
params_filename_value=$(func_parser_value "${lines[20]}")
save_file_key=$(func_parser_key "${lines[21]}")
save_file_value=$(func_parser_value "${lines[21]}")
opset_version_key=$(func_parser_key "${lines[22]}")
opset_version_value=$(func_parser_value "${lines[22]}")
enable_onnx_checker_key=$(func_parser_key "${lines[23]}")
enable_onnx_checker_value=$(func_parser_value "${lines[23]}")
paddle2onnx_params1_key=$(func_parser_key "${lines[24]}")
paddle2onnx_params1_value=$(func_parser_value "${lines[24]}")

# parser onnx inference 
inference_py=$(func_parser_value "${lines[25]}")
infer_cfg_key=$(func_parser_key "${lines[26]}")
onnx_file_key=$(func_parser_key "${lines[27]}")
infer_image_key=$(func_parser_key "${lines[28]}")
infer_image_value=$(func_parser_value "${lines[28]}")
infer_param1_key=$(func_parser_key "${lines[29]}")
infer_param1_value=$(func_parser_value "${lines[29]}")

LOG_PATH="./test_tipc/output/${model_name}/${MODE}"
mkdir -p ${LOG_PATH}
status_log="${LOG_PATH}/results_paddle2onnx.log"

function func_paddle2onnx_inference(){
    IFS='|'
    _python=$1
    _log_path=$2
    _export_model_dir=$3

    # paddle2onnx
    echo "################### run paddle2onnx ###################"
    set_dirname=$(func_set_params "${model_dir_key}" "${_export_model_dir}")
    set_model_filename=$(func_set_params "${model_filename_key}" "${model_filename_value}")
    set_params_filename=$(func_set_params "${params_filename_key}" "${params_filename_value}")
    set_save_model=$(func_set_params "${save_file_key}" "${_export_model_dir}/${save_file_value}")
    set_opset_version=$(func_set_params "${opset_version_key}" "${opset_version_value}")
    set_enable_onnx_checker=$(func_set_params "${enable_onnx_checker_key}" "${enable_onnx_checker_value}")
    set_paddle2onnx_params1=$(func_set_params "${paddle2onnx_params1_key}" "${paddle2onnx_params1_value}")
    trans_log_path="${_log_path}/trans_model.log"
    trans_model_cmd="${padlle2onnx_cmd} ${set_dirname} ${set_model_filename} ${set_params_filename} ${set_save_model} ${set_opset_version} ${set_enable_onnx_checker} ${set_paddle2onnx_params1}"
    eval "${trans_model_cmd} > ${trans_log_path} 2>&1"
    last_status=${PIPESTATUS[0]}
    cat ${trans_log_path}
    status_check $last_status "${trans_model_cmd}" "${status_log}" "${model_name}" "${trans_log_path}"

    # python inference
    echo "################### run onnx infer ###################"
    set_infer_cfg=$(func_set_params "${infer_cfg_key}" "${_export_model_dir}/infer_cfg.yml")
    set_onnx_file=$(func_set_params "${onnx_file_key}" "${_export_model_dir}/${save_file_value}")
    set_infer_image_file=$(func_set_params "${infer_image_key}" "${infer_image_value}")
    set_infer_param1=$(func_set_params "${infer_param1_key}" "${infer_param1_value}")
    _save_log_path="${_log_path}/paddle2onnx_infer_cpu.log"
    infer_model_cmd="${python} ${inference_py} ${set_infer_cfg} ${set_onnx_file} ${set_infer_image_file} ${set_infer_param1}"
    eval "${infer_model_cmd} > ${_save_log_path} 2>&1"
    last_status=${PIPESTATUS[0]}
    cat ${_save_log_path}
    status_check $last_status "${infer_model_cmd}" "${status_log}" "${model_name}" "${_save_log_path}"
}

export Count=0
IFS="|"
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
        set_export_param=$(func_set_params "${export_param_key}" "${export_param_value}")
        export_log_path="${LOG_PATH}/export.log"
        export_cmd="${python} ${run_export} ${set_export_weight} ${set_filename} ${set_export_param} ${set_save_export_dir} "
        echo  $export_cmd
        eval "${export_cmd} > ${export_log_path} 2>&1"
        status_export=$?
        cat ${export_log_path}
        status_check $status_export "${export_cmd}" "${status_log}" "${model_name}" "${export_log_path}"
    fi

    #run inference
    export_model_dir="${save_export_value}/${model_name}"
    func_paddle2onnx_inference "${python}" "${LOG_PATH}" "${export_model_dir}"
    Count=$(($Count + 1))
done
