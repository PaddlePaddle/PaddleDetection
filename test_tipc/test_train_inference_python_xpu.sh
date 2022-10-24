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

 # change gpu to xpu in tipc txt configs
 sed -i "s/use_gpu:True/use_xpu:True/g" $FILENAME
 sed -i "s/--device:gpu|cpu/--device:xpu|cpu/g" $FILENAME
 sed -i "s/trainer:pact_train/trainer:norm_train/g" $FILENAME
 sed -i "s/trainer:fpgm_train/trainer:norm_train/g" $FILENAME
 sed -i "s/--slim_config _template_pact/ /g" $FILENAME
 sed -i "s/--slim_config _template_fpgm/ /g" $FILENAME
 sed -i "s/--slim_config _template_kl_quant/ /g" $FILENAME
 sed -i 's/\"gpu\"/\"xpu\"/g' test_tipc/test_train_inference_python.sh

 # parser params
 dataline=`cat $FILENAME`
 IFS=$'\n'
 lines=(${dataline})

 # replace training config file
 grep -n '.yml' $FILENAME  | cut -d ":" -f 1 \
 | while read line_num ; do 
     train_cmd=$(func_parser_value "${lines[line_num-1]}")
     trainer_config=$(func_parser_config ${train_cmd})
     echo ${trainer_config}
     sed -i 's/use_gpu/use_xpu/g' "$REPO_ROOT_PATH/$trainer_config"
     # fine use_gpu in those included yaml
     sub_datalinee=`cat $REPO_ROOT_PATH/$trainer_config`
     IFS=$'\n'
     sub_lines=(${sub_datalinee})
     grep -n '.yml' "$REPO_ROOT_PATH/$trainer_config" | cut -d ":" -f 1 \
     | while read sub_line_num; do
         sub_config=${sub_lines[sub_line_num-1]} 
         dst=${#sub_config}-5
         sub_path=$(func_parser_dir "${trainer_config}")
         sub_config_path="${REPO_ROOT_PATH}${sub_path}/${sub_config:3:${dst}}"
         echo ${sub_config_path}
         sed -i 's/use_gpu/use_xpu/g' "$sub_config_path"
     done
 done

 # pass parameters to test_train_inference_python.sh
 cmd="bash test_tipc/test_train_inference_python.sh ${FILENAME} $2"
 echo $cmd
 eval $cmd