#-*-coding:utf-8-*-
# date:2021-12-20
# Author: Eric.Lee
## function: read datasets example

import os
import json
import cv2
from hand_data_iter.datasets import plot_box,draw_bd_handpose
import random

if __name__ == "__main__":
    path = "./handpose_datasets/"

    for f_ in os.listdir(path):
        if ".jpg" in f_:
            img_path = path +f_
            label_path = img_path.replace('.jpg','.json')
            if not os.path.exists(label_path):
                continue
            img_ = cv2.imread(img_path)

            f = open(label_path, encoding='utf-8')#读取 json文件
            hand_dict_ = json.load(f)
            f.close()

            hand_dict_ = hand_dict_["info"]
            print("len hand_dict :",len(hand_dict_))
            if len(hand_dict_)>0:
                for msg in hand_dict_:
                    bbox = msg["bbox"]
                    pts = msg["pts"]
                    print()
                    print(bbox)
                    RGB = (random.randint(50,255),random.randint(50,255),random.randint(50,255))
                    plot_box(bbox, img_, color=(RGB), label="hand", line_thickness=3)
                    draw_bd_handpose(img_,pts,bbox[0],bbox[1])

                    for k_ in pts.keys():
                        cv2.circle(img_, (int(pts[k_]['x']+bbox[0]),int(pts[k_]['y']+bbox[1])), 3, (255,50,155),-1)

                cv2.namedWindow("HandPose_Json",0)
                cv2.imshow("HandPose_Json",img_)
                cv2.waitKey(0)
