model_item=mask_rcnn_r50_1x_coco
bs_item=2
fp_item=fp32
run_mode=DP
device_num=N1C1
max_iter=100
num_workers=8

# get data
bash test_tipc/static/${model_item}/benchmark_common/prepare.sh
# run
bash test_tipc/static/${model_item}/benchmark_common/run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_mode} ${device_num} ${max_iter} ${num_workers} 2>&1;
# run profiling
sleep 10;
export PROFILING=true
bash test_tipc/static/${model_item}/benchmark_common/run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_mode} ${device_num} 11 ${num_workers} 2>&1;
