export FLAGS_npu_storage_format=0
export FLAGS_use_stride_kernel=0
export FLAGS_npu_jit_compile=0
export CUSTOM_DEVICE_BLACK_LIST=set_value,set_value_with_tensor
export ASCEND_GLOBAL_LOG_LEVEL=3

#指定npu
export ASCEND_RT_VISIBLE_DEVICES=14

# 启动测试
python tools/eval.py -c configs/solov2/solov2_r50_fpn_3x_coco.yml -o weights=output_cascade_rcnn/model_final.pdopt

