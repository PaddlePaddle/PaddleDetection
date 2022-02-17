model_item=mask_rcnn_r50_fpn_1x_coco
bs_item=2
fp_item=fp32
run_process_type=MultiP
run_mode=DP
device_num=N1C8
max_iter=500
num_workers=2

# get data
bash test_tipc/static/${model_item}/benchmark_common/prepare.sh
# run
bash test_tipc/static/${model_item}/benchmark_common/run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_process_type} ${run_mode} ${device_num} ${max_iter} ${num_workers} 2>&1;
