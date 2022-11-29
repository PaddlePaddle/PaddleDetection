#!/bin/bash
source test_tipc/utils_func.sh

FILENAME=$1
MODE="pipeline_infer"

# parser model_name
dataline=$(cat ${FILENAME})
IFS=$'\n'
lines=(${dataline})
model_name=$(func_parser_value "${lines[1]}")
echo "ppdet pipeline_python_infer: ${model_name}"
python=$(func_parser_value "${lines[2]}")
filename_key=$(func_parser_key "${lines[3]}")
filename_value=$(func_parser_value "${lines[3]}")

# parser infer params
infer_mode_list=$(func_parser_value "${lines[5]}")
input_key=$(func_parser_key "${lines[6]}")
input_list=$(func_parser_value "${lines[6]}")
use_gpu=$(func_parser_value "${lines[7]}")
inference_py=$(func_parser_value "${lines[8]}")
use_device_key=$(func_parser_key "${lines[9]}")
use_device_list=$(func_parser_value "${lines[9]}")
image_dir_key=$(func_parser_key "${lines[10]}")
infer_img_dir=$(func_parser_value "${lines[10]}")
video_dir_key=$(func_parser_key "${lines[11]}")
infer_video_dir=$(func_parser_value "${lines[11]}")


LOG_PATH="./test_tipc/output/${model_name}/${MODE}"
mkdir -p ${LOG_PATH}
status_log="${LOG_PATH}/results_serving_python.log"


function func_pipeline_inference(){
    IFS='|'
    _python=$1
    _log_path=$2
    _pipeline_script=$3
    _infer_dir=$4
    _input_type=$5
    _device_cmd=$6
    _device_type=$7
    # inference
   
    pipeline_log_path="${_log_path}/python_pipeline_${_input_type}_${_device_type}.log"
    output_path="--output_dir=${LOG_PATH}/"
    mot_flag="-o MOT.enable=True"
    if [ ${_input_type} = "video" ]; then
        pipeline_cmd="${_python} ${_pipeline_script} ${_infer_dir} ${_device_cmd} ${output_path} ${mot_flag} > ${pipeline_log_path} 2>&1 &"
    else
        pipeline_cmd="${_python} ${_pipeline_script} ${_infer_dir} ${_device_cmd} ${output_path}  > ${pipeline_log_path} 2>&1 &"
    fi
    # run 
    eval $pipeline_cmd
    last_status=${PIPESTATUS[0]}
    eval "cat ${pipeline_log_path}"
    status_check $last_status "${pipeline_cmd}" "${status_log}" "${model_name}" "${pipeline_log_path}"
       
}


#run  infer
Count=0
IFS="|"

for input in ${input_list[*]}; do
    for device_type in ${use_device_list[*]};do
        # set cuda device
        if [ ${use_gpu} = "False" ] || [ ${device_type} = "cpu" ]; then
            device_cmd=$(func_set_params "${use_device_key}" "${device_type}")
        elif [ ${use_gpu} = "True" ] && [ ${device_type} = "gpu" ]; then
            device_cmd=$(func_set_params "${use_device_key}" "${device_type}")
            env="export CUDA_VISIBLE_DEVICES=0"
            eval $env
        else
            echo "Does not support hardware other than CPU and GPU Currently!"
        fi
        if [ ${input} != "null" ]; then
            case ${input} in
                image) set_infer_file=$(func_set_params "${image_dir_key}" "${infer_img_dir}") ;;
                video) set_infer_file=$(func_set_params "${video_dir_key}" "${infer_video_dir}") ;;
                *) echo "Undefined input mode!"; exit 1;
            esac

        fi
        #run inference
        func_pipeline_inference "${python}" "${LOG_PATH}" "${inference_py}"  ${set_infer_file} ${input} ${device_cmd} ${device_type}
        Count=$(($Count + 1))
        eval "unset CUDA_VISIBLE_DEVICES"
    done
done

