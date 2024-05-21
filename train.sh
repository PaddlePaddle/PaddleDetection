export FLAGS_npu_storage_format=0
export FLAGS_use_stride_kernel=0
export FLAGS_npu_jit_compile=0
export CUSTOM_DEVICE_BLACK_LIST=set_value,set_value_with_tensor

#指定npu
# export ASCEND_RT_VISIBLE_DEVICES=11,12,13,14


# python -m paddle.distributed.launch --devices 0,1,2,3,4,5,6,7 --master=127.0.0.1:12345 tools/train.py -c configs/solov2/solov2_r50_fpn_3x_coco.yml -r output/23.pdparams  

# python -m paddle.distributed.launch --devices 8,9,10,11,12,13,14 --master=127.0.0.1:12347  --enable_ce tools/train.py -c configs/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.yml

ps -x | grep configs/cascade_rcnn | awk '{print $1}' | xargs kill -9
python -m paddle.distributed.launch --devices 8,9,10,11,12,13,14,15  tools/train.py -c configs/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.yml


# python tools/train.py -c configs/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.yml



# # 模型算子获取
# # 1) 执行模型训练之前，需要先输出以下环境变量
# export GLOG_v=6 # 新动态图执行器算子输出配置

# # 2) 执行模型训练，并 grep 日志文件，完成1-2次迭代之后即可停止
# # 启动单卡训练
# python tools/train.py -c configs/solov2/solov2_r50_fpn_1x_coco.yml > solov2.log 2>&1
# grep 日志的 loss 信息，确认训练已经完成 1-2 次迭代之后即可停止
# grep Loss solov2.log
# # 期望输出如下
# # [2022/07/05 10:38:20] ppcls INFO: [Train][Epoch 1/120][Iter: 0/5005]lr(PiecewiseDecay): 0.10000000, top1: 0.00000, top5: 0.00391, CELoss: 7.02506, loss: 7.02506, batch_cost: 6.29051s, reader_cost: 3.57217, ips: 40.69624 samples/s, eta: 43 days, 17:27:58
# # [2022/07/05 10:38:28] ppcls INFO: [Train][Epoch 1/120][Iter: 10/5005]lr(PiecewiseDecay): 0.10000000, top1: 0.00142, top5: 0.00604, CELoss: 8.36165, loss: 8.36165, batch_cost: 0.79642s, reader_cost: 0.02113, ips: 321.43838 samples/s, eta: 5 days, 12:52:01

# # 3) 执行如下命令从日志中筛选算子列表
# cat solov2.log | grep -a "API kernel key" | awk '{print $5}' > solov2_temp.log
# cat solov2.log | grep -a "API kernel key" | cut -d " " -f6 > solov2_temp.log
# sort -u solov2_temp.log > rsolov2_oplist.txt