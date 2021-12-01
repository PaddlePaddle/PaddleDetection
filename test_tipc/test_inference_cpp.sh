#!/bin/bash
source test_tipc/utils_func.sh

FILENAME=$1

# parser model_name
dataline=$(cat ${FILENAME})
IFS=$'\n'
lines=(${dataline})
model_name=$(func_parser_value "${lines[1]}")
echo "ppdet cpp_infer: ${model_name}"
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

# parser cpp inference model
opencv_dir=$(func_parser_value "${lines[15]}")
cpp_infer_mode_list=$(func_parser_value "${lines[16]}")
cpp_infer_is_quant_list=$(func_parser_value "${lines[17]}")
# parser cpp inference
inference_cmd=$(func_parser_value "${lines[18]}")
cpp_use_gpu_key=$(func_parser_key "${lines[19]}")
cpp_use_gpu_list=$(func_parser_value "${lines[19]}")
cpp_use_mkldnn_key=$(func_parser_key "${lines[20]}")
cpp_use_mkldnn_list=$(func_parser_value "${lines[20]}")
cpp_cpu_threads_key=$(func_parser_key "${lines[21]}")
cpp_cpu_threads_list=$(func_parser_value "${lines[21]}")
cpp_batch_size_key=$(func_parser_key "${lines[22]}")
cpp_batch_size_list=$(func_parser_value "${lines[22]}")
cpp_use_trt_key=$(func_parser_key "${lines[23]}")
cpp_use_trt_list=$(func_parser_value "${lines[23]}")
cpp_precision_key=$(func_parser_key "${lines[24]}")
cpp_precision_list=$(func_parser_value "${lines[24]}")
cpp_infer_model_key=$(func_parser_key "${lines[25]}")
cpp_image_dir_key=$(func_parser_key "${lines[26]}")
cpp_infer_img_dir=$(func_parser_value "${lines[26]}")
cpp_benchmark_key=$(func_parser_key "${lines[27]}")
cpp_benchmark_value=$(func_parser_value "${lines[27]}")
cpp_infer_key1=$(func_parser_key "${lines[28]}")
cpp_infer_value1=$(func_parser_value "${lines[28]}")

LOG_PATH="./test_tipc/output"
mkdir -p ${LOG_PATH}
status_log="${LOG_PATH}/results_cpp.log"

function func_cpp_inference(){
    IFS='|'
    _script=$1
    _model_dir=$2
    _log_path=$3
    _img_dir=$4
    _flag_quant=$5
    # inference
    for use_gpu in ${cpp_use_gpu_list[*]}; do
        if [ ${use_gpu} = "False" ] || [ ${use_gpu} = "cpu" ]; then
            for use_mkldnn in ${cpp_use_mkldnn_list[*]}; do
                if [ ${use_mkldnn} = "False" ] && [ ${_flag_quant} = "True" ]; then
                    continue
                fi
                for threads in ${cpp_cpu_threads_list[*]}; do
                    for batch_size in ${cpp_batch_size_list[*]}; do
                        _save_log_path="${_log_path}/cpp_infer_cpu_usemkldnn_${use_mkldnn}_threads_${threads}_precision_fluid_batchsize_${batch_size}.log"
                        set_infer_data=$(func_set_params "${cpp_image_dir_key}" "${_img_dir}")
                        set_benchmark=$(func_set_params "${cpp_benchmark_key}" "${cpp_benchmark_value}")
                        set_batchsize=$(func_set_params "${cpp_batch_size_key}" "${batch_size}")
                        set_cpu_threads=$(func_set_params "${cpp_cpu_threads_key}" "${threads}")
                        set_model_dir=$(func_set_params "${cpp_infer_model_key}" "${_model_dir}")
                        set_infer_params1=$(func_set_params "${cpp_infer_key1}" "${cpp_infer_value1}")
                        command="${_script} ${cpp_use_gpu_key}=${use_gpu} ${cpp_use_mkldnn_key}=${use_mkldnn} ${set_cpu_threads} ${set_model_dir} ${set_batchsize} ${set_infer_data} ${set_benchmark} ${set_infer_params1} > ${_save_log_path} 2>&1 "
                        eval $command
                        last_status=${PIPESTATUS[0]}
                        eval "cat ${_save_log_path}"
                        status_check $last_status "${command}" "${status_log}"
                    done
                done
            done
        elif [ ${use_gpu} = "True" ] || [ ${use_gpu} = "gpu" ]; then
            for precision in ${cpp_precision_list[*]}; do
                if [[ ${precision} != "fluid" ]]; then
                    if [[ ${_flag_quant} = "False" ]] && [[ ${precision} = "trt_int8" ]]; then
                        continue
                    fi
                    if [[ ${_flag_quant} = "True" ]] && [[ ${precision} != "trt_int8" ]]; then
                        continue
                    fi
                fi
                for batch_size in ${cpp_batch_size_list[*]}; do
                    _save_log_path="${_log_path}/cpp_infer_gpu_precision_${precision}_batchsize_${batch_size}.log"
                    set_infer_data=$(func_set_params "${cpp_image_dir_key}" "${_img_dir}")
                    set_benchmark=$(func_set_params "${cpp_benchmark_key}" "${cpp_benchmark_value}")
                    set_batchsize=$(func_set_params "${cpp_batch_size_key}" "${batch_size}")
                    set_precision=$(func_set_params "${cpp_precision_key}" "${precision}")
                    set_model_dir=$(func_set_params "${cpp_infer_model_key}" "${_model_dir}")
                    set_infer_params1=$(func_set_params "${cpp_infer_key1}" "${cpp_infer_value1}")
                    command="${_script} ${cpp_use_gpu_key}=${use_gpu} ${set_precision} ${set_model_dir} ${set_batchsize} ${set_infer_data} ${set_benchmark} ${set_infer_params1} > ${_save_log_path} 2>&1 "
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

cd ./deploy/cpp
# set OPENCV_DIR
if [ ${opencv_dir} = "default" ] || [ ${opencv_dir} = "null" ]; then
    OPENCV_DIR=$(pwd)/deps/opencv-3.4.16_gcc8.2_ffmpeg
else
    OPENCV_DIR=${opencv_dir}
fi

# build program
# TODO: set PADDLE_DIR and TENSORRT_ROOT
if [ -z $PADDLE_DIR ]; then
    PADDLE_DIR=/paddle/Paddle/build/paddle_inference_install_dir/
fi
if [ -z $TENSORRT_ROOT ]; then
    TENSORRT_ROOT=/usr/local/TensorRT6-cuda10.1-cudnn7
fi
CUDA_LIB=$(dirname `find /usr -name libcudart.so`)
CUDNN_LIB=$(dirname `find /usr -name libcudnn.so`)
TENSORRT_LIB_DIR="${TENSORRT_ROOT}/lib"
TENSORRT_INC_DIR="${TENSORRT_ROOT}/include"

rm -rf build
mkdir -p build
cd ./build
cmake .. \
    -DWITH_GPU=ON \
    -DWITH_MKL=ON \
    -DWITH_TENSORRT=ON \
    -DPADDLE_LIB_NAME=libpaddle_inference \
    -DPADDLE_DIR=${PADDLE_DIR} \
    -DCUDA_LIB=${CUDA_LIB} \
    -DCUDNN_LIB=${CUDNN_LIB} \
    -DTENSORRT_LIB_DIR=${TENSORRT_LIB_DIR} \
    -DTENSORRT_INC_DIR=${TENSORRT_INC_DIR} \
    -DOPENCV_DIR=${OPENCV_DIR} \
    -DWITH_KEYPOINT=ON \
    -DWITH_MOT=ON

make -j4
cd ../../../
echo "################### build finished! ###################"


# set cuda device
GPUID=$2
if [ ${#GPUID} -le 0 ];then
    env=" "
else
    env="export CUDA_VISIBLE_DEVICES=${GPUID}"
fi
eval $env
# run cpp infer
Count=0
IFS="|"
infer_quant_flag=(${cpp_infer_is_quant_list})
for infer_mode in ${cpp_infer_mode_list[*]}; do

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
    set_filename=$(func_set_params "${filename_key}" "${filename_value}")
    export_cmd="${python} ${run_export} ${set_export_weight} ${set_filename} ${set_save_export_dir} "
    echo  $export_cmd
    eval $export_cmd
    status_export=$?
    status_check $status_export "${export_cmd}" "${status_log}"

    #run inference
    save_export_model_dir="${save_export_value}/${filename_value}"
    is_quant=${infer_quant_flag[Count]}
    func_cpp_inference "${inference_cmd}" "${save_export_model_dir}" "${LOG_PATH}" "${cpp_infer_img_dir}" ${is_quant}
    Count=$(($Count + 1))
done
eval "unset CUDA_VISIBLE_DEVICES"
