data_path=/paddle/dataset/bdd100kmot/bdd100k_path
img_dir=${data_path}/images
label_dir=${data_path}/labels
save_path=${data_path}/bdd100k_vehicle

phasetrain=train
phaseval=val
classes=2,3,4,9,10

# 生成mot格式的数据
python bdd2mot.py --img_dir=${img_dir} --label_dir=${label_dir} --save_path=${save_path}

python bdd100k2mot.py --data_path=${data_path}  --phase=${phasetrain} --classes=${classes}
python bdd100k2mot.py --data_path=${data_path} --phase=${phaseval} --classes=${classes}

# 生成新的单类别的数据
python gen_labels_MOT.py --mot_data=${data_path} --phase=${phasetrain}
python gen_labels_MOT.py --mot_data=${data_path} --phase=${phaseval}