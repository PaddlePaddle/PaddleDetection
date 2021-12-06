# Use docker: paddlepaddle/paddle:latest-gpu-cuda10.1-cudnn7  paddle=2.1.2  python3.7
#
# Usage:
#   git clone https://github.com/PaddlePaddle/PaddleDetection.git
#   cd PaddleDetection
#   bash benchmark/run_all.sh
log_path=${LOG_PATH_INDEX_DIR:-$(pwd)}  #  benchmark系统指定该参数,不需要跑profile时,log_path指向存speed的目录

# run prepare.sh
bash benchmark/prepare.sh

model_name_list=(faster_rcnn fcos deformable_detr gfl hrnet higherhrnet solov2 jde fairmot)
fp_item_list=(fp32)
max_epoch=2

for model_item in ${model_name_list[@]}; do
      for fp_item in ${fp_item_list[@]}; do
          case ${model_item} in
              faster_rcnn) bs_list=(1 8) ;;
              fcos) bs_list=(2) ;;
              deformable_detr) bs_list=(2) ;;
              gfl) bs_list=(2) ;;
              hrnet) bs_list=(64) ;;
              higherhrnet) bs_list=(20) ;;
              solov2) bs_list=(2) ;;
              jde) bs_list=(4) ;;
              fairmot) bs_list=(6) ;;
              *) echo "wrong model_name"; exit 1;
          esac
          for bs_item in ${bs_list[@]}
            do
            run_mode=sp
            log_name=detection_${model_item}_bs${bs_item}_${fp_item}   # 如:clas_MobileNetv1_mp_bs32_fp32_8
            echo "index is speed, 1gpus, begin, ${log_name}"
            CUDA_VISIBLE_DEVICES=0 bash benchmark/run_benchmark.sh ${run_mode} ${bs_item} \
             ${fp_item} ${max_epoch} ${model_item} | tee ${log_path}/${log_name}_speed_1gpus 2>&1
            sleep 60

            run_mode=mp
            log_name=detection_${model_item}_bs${bs_item}_${fp_item}   # 如:clas_MobileNetv1_mp_bs32_fp32_8
            echo "index is speed, 8gpus, run_mode is multi_process, begin, ${log_name}"
            CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash benchmark/run_benchmark.sh ${run_mode} \
             ${bs_item} ${fp_item} ${max_epoch} ${model_item}| tee ${log_path}/${log_name}_speed_8gpus8p 2>&1
            sleep 60
            done
      done
done
