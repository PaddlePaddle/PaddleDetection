# Use docker: paddlepaddle/paddle:latest-gpu-cuda10.1-cudnn7  paddle=2.1.2  python3.7
#
# Usage:
#   git clone https://github.com/PaddlePaddle/PaddleDetection.git
#   cd PaddleDetection
#   bash benchmark/run_all.sh

# run prepare.sh
bash benchmark/prepare.sh

model_name_list=(faster_rcnn fcos deformable_detr gfl)
fp_item_list=(fp32)
max_epoch=1

for model_name in ${model_name_list[@]}; do
      for fp_item in ${fp_item_list[@]}; do
          case ${model_name} in
              faster_rcnn) bs_list=(1 8) ;;
              fcos) bs_list=(2 8) ;;
              deformable_detr) bs_list=(2) ;;
              gfl) bs_list=(2 8) ;;
              *) echo "wrong model_name"; exit 1;
          esac
          for bs_item in ${bs_list[@]}
            do
            echo "index is speed, 1gpus, begin, ${model_name}"
            run_mode=sp
            CUDA_VISIBLE_DEVICES=0 bash benchmark/run_benchmark.sh ${run_mode} ${bs_item} \
             ${fp_item} ${max_epoch} ${model_name}     #  (5min)
            sleep 60

            echo "index is speed, 8gpus, run_mode is multi_process, begin, ${model_name}"
            run_mode=mp
            CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash benchmark/run_benchmark.sh ${run_mode} \
             ${bs_item} ${fp_item} ${max_epoch} ${model_name}
            sleep 60
            done
      done
done
