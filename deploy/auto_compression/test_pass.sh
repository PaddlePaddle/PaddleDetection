#!/bin/bash

# 此脚本可以逐个删除pass，然后运行测试脚本，检查去掉该pass是否对模型精度有影响
pass_array=("mkldnn_placement_pass" "simplify_with_basic_ops_pass"
        "layer_norm_fuse_pass"
        "attention_lstm_fuse_pass"
        "seqconv_eltadd_relu_fuse_pass"
        "seqpool_cvm_concat_fuse_pass"
        "mul_lstm_fuse_pass"
        "fc_gru_fuse_pass"
        "mul_gru_fuse_pass"
        "seq_concat_fc_fuse_pass"
        "gpu_cpu_squeeze2_matmul_fuse_pass"
        "gpu_cpu_reshape2_matmul_fuse_pass"
        "gpu_cpu_flatten2_matmul_fuse_pass"
        "matmul_v2_scale_fuse_pass"
        "gpu_cpu_map_matmul_v2_to_mul_pass"
        "gpu_cpu_map_matmul_v2_to_matmul_pass"
        "matmul_scale_fuse_pass"
        "gpu_cpu_map_matmul_to_mul_pass"
        "fc_fuse_pass"
        "repeated_fc_relu_fuse_pass"
        "squared_mat_sub_fuse_pass"
        "conv_bn_fuse_pass"
        "conv_eltwiseadd_bn_fuse_pass"
        "conv_transpose_bn_fuse_pass"
        "conv_transpose_eltwiseadd_bn_fuse_pass"
        "is_test_pass"
        "constant_folding_pass"
        "squeeze2_transpose2_onednn_fuse_pass"
        "depthwise_conv_mkldnn_pass"
        "conv_bn_fuse_pass"
        "conv_eltwiseadd_bn_fuse_pass"
        "conv_affine_channel_mkldnn_fuse_pass"
        "conv_transpose_bn_fuse_pass"
        "conv_transpose_eltwiseadd_bn_fuse_pass"
        "conv_bias_mkldnn_fuse_pass"
        "conv_transpose_bias_mkldnn_fuse_pass"
        "conv_elementwise_add_mkldnn_fuse_pass"
        "conv_activation_mkldnn_fuse_pass"
        "scale_matmul_fuse_pass"
        "reshape_transpose_matmul_mkldnn_fuse_pass"
        "matmul_transpose_reshape_mkldnn_fuse_pass"
        "matmul_elementwise_add_mkldnn_fuse_pass"
        "matmul_activation_mkldnn_fuse_pass"
        "fc_mkldnn_pass"
        "fc_act_mkldnn_fuse_pass"
        "fc_elementwise_add_mkldnn_fuse_pass"
        "batch_norm_act_fuse_pass"
        "softplus_activation_onednn_fuse_pass"
        "shuffle_channel_mkldnn_detect_pass"
        "elementwise_act_onednn_fuse_pass"
        "layer_norm_onednn_optimization_pass"
        "operator_scale_onednn_fuse_pass"
        "operator_unsqueeze2_onednn_fuse_pass"
        "operator_reshape2_onednn_fuse_pass") 

pass_count=${#pass_array[@]}

# 定义日志文件路径
log_file="test_mkldnn_pass.txt"

# 循环执行测试脚本
for ((i=0; i<pass_count; i++)); do
  current_pass=${pass_array[$i]}
  pass_log_file="${current_pass}_log.txt"
  echo "Delete pass: $current_pass"
  echo "Delete pass: $current_pass" >> "$log_file"
  python test_det.py --model_path=models/dino_dir/dino_r50_4scale_2x_coco --config=configs/dino_reader.yml --precision=fp32 --use_mkldnn=True --device=CPU --cpu_threads=24 --delete_pass_name "$current_pass" >> "$pass_log_file" 2>&1

  if [ $? -eq 0 ]; then
    echo "python executed successfully after delete pass $current_pass"
    cat "$pass_log_file" >> "$log_file"
  else
    echo "Error occurred during delete pass($current_pass) execution"
    echo "Error occurred during delete pass($current_pass) execution" >> "$log_file"
    cat "$pass_log_file" >> "$log_file"
  fi

  rm "$pass_log_file"
done