data_path=bdd100k
img_dir=${data_path}/images/track
label_dir=${data_path}/labels/box_track_20
save_path=${data_path}/bdd100kmot_vehicle

phasetrain=train
phaseval=val
classes=2,3,4,9,10

# gen mot dataste
python bdd100k2mot.py --data_path=${data_path}  --phase=${phasetrain} --classes=${classes} --img_dir=${img_dir} --label_dir=${label_dir} --save_path=${save_path}
python bdd100k2mot.py --data_path=${data_path} --phase=${phaseval} --classes=${classes} --img_dir=${img_dir} --label_dir=${label_dir} --save_path=${save_path}

# gen new labels_with_ids
python gen_labels_MOT.py --mot_data=${data_path} --phase=${phasetrain}
python gen_labels_MOT.py --mot_data=${data_path} --phase=${phaseval}
