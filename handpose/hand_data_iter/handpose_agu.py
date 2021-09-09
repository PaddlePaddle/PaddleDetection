#-*-coding:utf-8-*-
# date:2021-12-20
# Author: Eric.Lee
## function: handpose agu

import json
import cv2
import os
import random
from data_agu import hand_alignment_aug_fun
import numpy as np

def plot_box(bbox, img, color=None, label=None, line_thickness=None):
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)# 目标的bbox
    if label:
        tf = max(tl - 2, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 4, thickness=tf)[0] # label size
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3 # 字体的bbox
        cv2.rectangle(img, c1, c2, color, -1)  # label 矩形填充
        # 文本绘制
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 4, [225, 255, 255],thickness=tf, lineType=cv2.LINE_AA)
def draw_bd_handpose(img_,hand_,x,y):
    thick = 2
    colors = [(0,215,255),(255,115,55),(5,255,55),(25,15,255),(225,15,55)]
    #
    cv2.line(img_, (int(hand_['0']['x']+x), int(hand_['0']['y']+y)),(int(hand_['1']['x']+x), int(hand_['1']['y']+y)), colors[0], thick)
    cv2.line(img_, (int(hand_['1']['x']+x), int(hand_['1']['y']+y)),(int(hand_['2']['x']+x), int(hand_['2']['y']+y)), colors[0], thick)
    cv2.line(img_, (int(hand_['2']['x']+x), int(hand_['2']['y']+y)),(int(hand_['3']['x']+x), int(hand_['3']['y']+y)), colors[0], thick)
    cv2.line(img_, (int(hand_['3']['x']+x), int(hand_['3']['y']+y)),(int(hand_['4']['x']+x), int(hand_['4']['y']+y)), colors[0], thick)

    cv2.line(img_, (int(hand_['0']['x']+x), int(hand_['0']['y']+y)),(int(hand_['5']['x']+x), int(hand_['5']['y']+y)), colors[1], thick)
    cv2.line(img_, (int(hand_['5']['x']+x), int(hand_['5']['y']+y)),(int(hand_['6']['x']+x), int(hand_['6']['y']+y)), colors[1], thick)
    cv2.line(img_, (int(hand_['6']['x']+x), int(hand_['6']['y']+y)),(int(hand_['7']['x']+x), int(hand_['7']['y']+y)), colors[1], thick)
    cv2.line(img_, (int(hand_['7']['x']+x), int(hand_['7']['y']+y)),(int(hand_['8']['x']+x), int(hand_['8']['y']+y)), colors[1], thick)

    cv2.line(img_, (int(hand_['0']['x']+x), int(hand_['0']['y']+y)),(int(hand_['9']['x']+x), int(hand_['9']['y']+y)), colors[2], thick)
    cv2.line(img_, (int(hand_['9']['x']+x), int(hand_['9']['y']+y)),(int(hand_['10']['x']+x), int(hand_['10']['y']+y)), colors[2], thick)
    cv2.line(img_, (int(hand_['10']['x']+x), int(hand_['10']['y']+y)),(int(hand_['11']['x']+x), int(hand_['11']['y']+y)), colors[2], thick)
    cv2.line(img_, (int(hand_['11']['x']+x), int(hand_['11']['y']+y)),(int(hand_['12']['x']+x), int(hand_['12']['y']+y)), colors[2], thick)

    cv2.line(img_, (int(hand_['0']['x']+x), int(hand_['0']['y']+y)),(int(hand_['13']['x']+x), int(hand_['13']['y']+y)), colors[3], thick)
    cv2.line(img_, (int(hand_['13']['x']+x), int(hand_['13']['y']+y)),(int(hand_['14']['x']+x), int(hand_['14']['y']+y)), colors[3], thick)
    cv2.line(img_, (int(hand_['14']['x']+x), int(hand_['14']['y']+y)),(int(hand_['15']['x']+x), int(hand_['15']['y']+y)), colors[3], thick)
    cv2.line(img_, (int(hand_['15']['x']+x), int(hand_['15']['y']+y)),(int(hand_['16']['x']+x), int(hand_['16']['y']+y)), colors[3], thick)

    cv2.line(img_, (int(hand_['0']['x']+x), int(hand_['0']['y']+y)),(int(hand_['17']['x']+x), int(hand_['17']['y']+y)), colors[4], thick)
    cv2.line(img_, (int(hand_['17']['x']+x), int(hand_['17']['y']+y)),(int(hand_['18']['x']+x), int(hand_['18']['y']+y)), colors[4], thick)
    cv2.line(img_, (int(hand_['18']['x']+x), int(hand_['18']['y']+y)),(int(hand_['19']['x']+x), int(hand_['19']['y']+y)), colors[4], thick)
    cv2.line(img_, (int(hand_['19']['x']+x), int(hand_['19']['y']+y)),(int(hand_['20']['x']+x), int(hand_['20']['y']+y)), colors[4], thick)

if __name__ == "__main__":
    path = "/handpose_datasets/"
    vis = True
    hand_idx = 0
    for f_ in os.listdir(path):
        if ".jpg" in f_:
            img_path = path +f_
            label_path = img_path.replace('.jpg','.json')
            if not os.path.exists(label_path):
                continue
            img_ = cv2.imread(img_path)
            img_ago = img_.copy()
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
                    # print(pts)
                    x1,y1,x2,y2 = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
                    hand = img_ago[y1:y2,x1:x2,:]
                    pts_ = []

                    x_max = -65535
                    y_max = -65535
                    x_min = 65535
                    y_min = 65535
                    for i in range(21):
                        x_,y_ = pts[str(i)]["x"],pts[str(i)]["y"]
                        x_ += x1
                        y_ += y1
                        pts_.append([x_,y_])
                        x_min = x_ if x_min>x_ else x_min
                        y_min = y_ if y_min>y_ else y_min
                        x_max = x_ if x_max<x_ else x_max
                        y_max = y_ if y_max<y_ else y_max
                    if vis:
                        plot_box((x_min,y_min,x_max,y_max), img_, color=(255,100,100), label="hand", line_thickness=2)
                    #
                    if True:
                        angle_random = random.randint(-22,22)

                        offset_x = int((x_max-x_min)/8)
                        offset_y = int((y_max-y_min)/8)


                        pt_left = (x_min+random.randint(-offset_x,offset_x),(y_min+y_max)/2+random.randint(-offset_y,offset_y))
                        pt_right = (x_max+random.randint(-offset_x,offset_x),(y_min+y_max)/2+random.randint(-offset_y,offset_y))
                        angle_random = random.randint(-90,90)
                        scale_x = float(random.randint(20,40))/100.

                        hand_rot,pts_tor_landmarks,M_I = hand_alignment_aug_fun(img_ago,pt_left,pt_right,
                            facial_landmarks_n = pts_,\
                            angle = angle_random,desiredLeftEye=(scale_x, 0.5),
                            desiredFaceWidth=256, desiredFaceHeight=None,draw_flag = True)
                    else:

                        offset_x = 0
                        offset_y = 0

                        pt_left = (x_min+random.randint(-offset_x,offset_x),(y_min+y_max)/2+random.randint(-offset_y,offset_y))
                        pt_right = (x_max+random.randint(-offset_x,offset_x),(y_min+y_max)/2+random.randint(-offset_y,offset_y))
                        angle_random = 0
                        scale_x = 0.25

                        hand_rot,pts_tor_landmarks,M_I = hand_alignment_aug_fun(img_ago,pt_left,pt_right,
                            facial_landmarks_n = pts_,\
                            angle = angle_random,desiredLeftEye=(scale_x, 0.5),
                            desiredFaceWidth=256, desiredFaceHeight=None,draw_flag = True)
                    #
                    hand_idx += 1
                    cv2.imwrite("../test_datasets/{}.jpg".format(hand_idx),hand_rot)
                    #
                    pts_hand = {}
                    pts_hand_global_rot = {}
                    for ptk in range(21):
                        xh,yh = pts_tor_landmarks[ptk][0],pts_tor_landmarks[ptk][1]
                        pts_hand[str(ptk)] = {}
                        pts_hand[str(ptk)] = {
                            "x":xh,
                            "y":yh,
                            }
                        #------------
                        x_r = (xh*M_I[0][0] + yh*M_I[0][1] + M_I[0][2])
                        y_r = (xh*M_I[1][0] + yh*M_I[1][1] + M_I[1][2])

                        pts_hand_global_rot[str(ptk)] = {
                            "x":x_r,
                            "y":y_r,
                            }

                    if vis:
                        draw_bd_handpose(hand_rot,pts_hand,0,0)
                        cv2.namedWindow("hand_rot",0)
                        cv2.imshow("hand_rot",hand_rot)

                        cv2.namedWindow("hand_origin",0)
                        cv2.imshow("hand_origin",hand)
                        RGB = (random.randint(50,255),random.randint(50,255),random.randint(50,255))
                        plot_box((x1,y1,x2,y2), img_, color=(RGB), label="hand", line_thickness=3)
                        # draw_bd_handpose(img_,pts,bbox[0],bbox[1])
                        draw_bd_handpose(img_,pts_hand_global_rot,0,0)
                if vis:
                    cv2.putText(img_, 'len:{}'.format(len(hand_dict_)), (5,40),
                                cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0),4)
                    cv2.putText(img_, 'len:{}'.format(len(hand_dict_)), (5,40),
                                cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255))
                    cv2.namedWindow("Gesture_json",0)
                    cv2.imshow("Gesture_json",img_)
                    if cv2.waitKey(1) == 27:
                        break
