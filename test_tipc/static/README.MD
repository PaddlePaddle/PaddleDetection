# PaddleDetection 下静态图benchmark模型执行说明
静态图benchmark测试脚本说明
# 目录说明 
# Docker 运行环境
docker image: registry.baidubce.com/paddlepaddle/paddle:latest-dev-cuda11.2-cudnn8-gcc82
paddle = 2.2.2
python = 3.7
# 运行benchmark测试步骤
git clone https://github.com/PaddlePaddle/PaddleDetection.git
cd PaddleDetection
# 准备数据
bash test_tipc/static/${model_item}/benchmark_common/prepare.sh
# 运行模型
## 单卡（自动运行打开Profiling）
export CUDA_VISIBLE_DEVICES=0 
bash  test_tipc/static/${model_item}/N1C1/yolov3_darknet53_270e_coco_bs8_fp32_SingleP_DP.sh 
## 多卡
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
bash test_tipc/static/${model_item}/N1C8/yolov3_darknet53_270e_coco_bs8_fp32_MultiP_DP.sh
