#!/bin/bash
source test_tipc/utils_func.sh
function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}
function func_parser_config() {
    strs=$1
    IFS=" "
    array=(${strs})
    tmp=${array[2]}
    echo ${tmp}
}
function func_parser_dir() {
    strs=$1
    IFS="/"
    array=(${strs})
    len=${#array[*]}
    dir=""
    count=1
    for arr in ${array[*]}; do 
        if [ ${len} = "${count}" ]; then
            continue;
        else
            dir="${dir}/${arr}"
            count=$((${count} + 1))
        fi
    done
    echo "${dir}"
}
BASEDIR=$(dirname "$0")
REPO_ROOT_PATH=$(readlinkf ${BASEDIR}/../)
FILENAME=$1
 # change gpu to npu in tipc txt configs
 sed -i "s/use_gpu:True/use_npu:True/g" $FILENAME
 sed -i "s/--device:gpu|cpu/--device:npu|cpu/g" $FILENAME
 sed -i "s/--device:gpu/--device:npu/g" $FILENAME
 sed -i "s/--device:cpu|gpu/--device:cpu|npu/g" $FILENAME
 sed -i "s/trainer:pact_train/trainer:norm_train/g" $FILENAME
 sed -i "s/trainer:fpgm_train/trainer:norm_train/g" $FILENAME
 sed -i "s/--slim_config _template_pact/ /g" $FILENAME
 sed -i "s/--slim_config _template_fpgm/ /g" $FILENAME
 sed -i "s/--slim_config _template_kl_quant/ /g" $FILENAME
 sed -i 's/\"gpu\"/\"npu\"/g' test_tipc/test_train_inference_python.sh

 # parser params
dataline=`cat $FILENAME`
IFS=$'\n'
lines=(${dataline})
# replace training config file
grep -n '.yml' $FILENAME  | cut -d ":" -f 1 \
| while read line_num ; do 
    train_cmd=$(func_parser_value "${lines[line_num-1]}")
    trainer_config=$(func_parser_config ${train_cmd})
    sed -i 's/use_gpu/use_npu/g' "$REPO_ROOT_PATH/$trainer_config"
    sed -i 's/aligned: True/aligned: False/g' "$REPO_ROOT_PATH/$trainer_config"
    # fine use_gpu in those included yaml
    sub_datalinee=`cat $REPO_ROOT_PATH/$trainer_config`
    IFS=$'\n'
    sub_lines=(${sub_datalinee})
    grep -n '.yml' "$REPO_ROOT_PATH/$trainer_config" | cut -d ":" -f 1 \
    | while read sub_line_num; do
        sub_config=${sub_lines[sub_line_num-1]} 
        dst=${#sub_config}-5
        sub_path=$(func_parser_dir "${trainer_config}")
        sub_config_name=$(echo "$sub_config" | awk -F"'" '{ print $2 }')
        sub_config_path="${REPO_ROOT_PATH}${sub_path}/${sub_config_name}"
        sed -i 's/use_gpu/use_npu/g' "$sub_config_path"
        sed -i 's/aligned: True/aligned: False/g' "$sub_config_path"
    done
done


# NPU lacks operators such as deformable_conv, depthwise_conv2d_transpose, 
# which will affects ips. Here, we reduce the number of coco training sets 
# for npu tipc bencnmark. This is a temporary hack.
# # TODO(duanyanhui): add vision ops for npu 
train_img_num=`cat $REPO_ROOT_PATH/dataset/coco/annotations/instances_train2017.json | grep -o  file_name | wc -l`
exp_num=8
if [ ${train_img_num} != ${exp_num} ];then
    echo "Replace with npu tipc coco training annotations"
    mv $REPO_ROOT_PATH/dataset/coco/annotations/instances_train2017.json $REPO_ROOT_PATH/dataset/coco/annotations/instances_train2017_bak.json
    wget https://paddle-device.bj.bcebos.com/tipc/instances_train2017.json
    mv instances_train2017.json $REPO_ROOT_PATH/dataset/coco/annotations/
    rm -f instances_train2017.json
fi

# pass parameters to test_train_inference_python.sh
cmd="bash test_tipc/test_train_inference_python.sh ${FILENAME} $2"
echo $cmd
eval $cmd
