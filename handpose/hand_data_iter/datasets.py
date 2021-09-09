#-*-coding:utf-8-*-
# date:2019-05-20
# Author: Eric.Lee
# function: data iter
import glob
import math
import os
import random
import shutil
from pathlib import Path
from PIL import Image
# import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import numpy as np
from paddle.io import Dataset
from hand_data_iter.data_agu import *

import shutil
import json

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

def img_agu_channel_same(img_):
    img_a = np.zeros(img_.shape, dtype = np.uint8)
    gray = cv2.cvtColor(img_,cv2.COLOR_RGB2GRAY)
    img_a[:,:,0] =gray
    img_a[:,:,1] =gray
    img_a[:,:,2] =gray

    return img_a
# 图像白化
def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y

# 图像亮度、对比度增强
def contrast_img(img, c, b):  # 亮度就是每个像素所有通道都加上b
    rows, cols, channels = img.shape
    # 新建全零(黑色)图片数组:np.zeros(img1.shape, dtype=uint8)
    blank = np.zeros([rows, cols, channels], img.dtype)
    dst = cv2.addWeighted(img, c, blank, 1-c, b)
    return dst

class LoadImagesAndLabels(Dataset):
    def __init__(self, ops, img_size=(224,224), flag_agu = False,fix_res = True,vis = False):

        # vis = True
        print('img_size (height,width) : ',img_size[0],img_size[1])
        print("train_path : {}".format(ops.train_path))

        path = ops.train_path

        file_list = []
        hand_anno_list = []
        idx = 0
        for f_ in os.listdir(path):
            if ".jpg" in f_:
                img_path = path +f_
                label_path = img_path.replace('.jpg','.json')
                if not os.path.exists(label_path):
                    continue

                f = open(label_path, encoding='utf-8')#读取 json文件
                hand_dict_ = json.load(f)
                f.close()
                if len(hand_dict_)==0:
                    continue

                hand_dict_ = hand_dict_["info"]

                #----------------------------------------------
                if vis:
                    img_ = cv2.imread(img_path)
                    img_ago = img_.copy()

                    # cv2.namedWindow("hand_d",0)
                    # cv2.imshow("hand_d",img_ago)
                    # cv2.waitKey(1)
                #----------------------------------------------
                # print("len hand_dict :",len(hand_dict_))
                if len(hand_dict_)>0:
                    for msg in hand_dict_:
                        bbox = msg["bbox"]
                        pts = msg["pts"]
                        file_list.append(img_path)
                        hand_anno_list.append((bbox,pts))
                        idx += 1
                        print("  hands num : {}".format(idx),end = "\r")
                        #------------------------------------
                        if vis:
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

                            plot_box((x_min,y_min,x_max,y_max), img_, color=(255,100,100), label="hand", line_thickness=2)


                            offset_x = int((x_max-x_min)/8)
                            offset_y = int((y_max-y_min)/8)


                            pt_left = (x_min+random.randint(-offset_x,offset_x),(y_min+y_max)/2+random.randint(-offset_y,offset_y))
                            pt_right = (x_max+random.randint(-offset_x,offset_x),(y_min+y_max)/2+random.randint(-offset_y,offset_y))
                            angle_random = random.randint(-180,180)
                            scale_x = float(random.randint(20,32))/100.
                            hand_rot,pts_tor_landmarks,_ = hand_alignment_aug_fun(img_ago,pt_left,pt_right,
                                facial_landmarks_n = pts_,\
                                angle = angle_random,desiredLeftEye=(scale_x, 0.5),
                                desiredFaceWidth=img_size[0], desiredFaceHeight=None,draw_flag = True)

                            pts_hand = {}
                            for ptk in range(21):
                                xh,yh = pts_tor_landmarks[ptk][0],pts_tor_landmarks[ptk][1]
                                pts_hand[str(ptk)] = {}
                                pts_hand[str(ptk)] = {
                                    "x":xh,
                                    "y":yh,
                                    }

                            draw_bd_handpose(hand_rot,pts_hand,0,0)# 绘制关键点 连线

                            cv2.namedWindow("hand_rotd",0)
                            cv2.imshow("hand_rotd",hand_rot)
                            print("hand_rot shape : {}".format(hand_rot.shape))
                            cv2.waitKey(1)


        #
        print()
        self.files = file_list
        self.hand_anno_list = hand_anno_list
        self.img_size = img_size
        self.flag_agu = flag_agu
        # self.fix_res = fix_res
        self.vis = vis

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = self.files[index]
        bbox,pts = self.hand_anno_list[index]
        img = cv2.imread(img_path)  # BGR
        #-------------------------------------
        x1,y1,x2,y2 = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])

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

        if random.random() > 0.55:
            offset_x = int((x_max-x_min)/8)
            offset_y = int((y_max-y_min)/8)

            pt_left = (x_min+random.randint(-offset_x,offset_x),(y_min+y_max)/2+random.randint(-offset_y,offset_y))
            pt_right = (x_max+random.randint(-offset_x,offset_x),(y_min+y_max)/2+random.randint(-offset_y,offset_y))
            angle_random = random.randint(-180,180)
            scale_x = float(random.randint(12,33))/100.
            hand_rot,pts_tor_landmarks,_ = hand_alignment_aug_fun(img,pt_left,pt_right,
                facial_landmarks_n = pts_,\
                angle = angle_random,desiredLeftEye=(scale_x, 0.5),
                desiredFaceWidth=self.img_size[0], desiredFaceHeight=None,draw_flag = False)
            if self.vis:
                pts_hand = {}
                for ptk in range(21):
                    xh,yh = pts_tor_landmarks[ptk][0],pts_tor_landmarks[ptk][1]
                    pts_hand[str(ptk)] = {}
                    pts_hand[str(ptk)] = {
                        "x":xh,
                        "y":yh,
                        }

                draw_bd_handpose(hand_rot,pts_hand,0,0)
                cv2.namedWindow("hand_rotd",0)
                cv2.imshow("hand_rotd",hand_rot)
                cv2.waitKey(1)

            img_ = hand_rot
            pts_tor_landmarks_norm = []
            for i in range(len(pts_tor_landmarks)):
                x_ = float(pts_tor_landmarks[i][0])/float(self.img_size[0])
                y_ = float(pts_tor_landmarks[i][1])/float(self.img_size[0])
                pts_tor_landmarks_norm.append([x_,y_])

        else:
            w_ = max(abs(x_max-x_min),abs(y_max-y_min))
            w_ = w_*(1.+float(random.randint(5,40))/100.)
            x_mid = (x_max+x_min)/2
            y_mid = (y_max+y_min)/2

            x1,y1,x2,y2 = int(x_mid-w_/2.),int(y_mid-w_/2.),int(x_mid+w_/2.),int(y_mid+w_/2.)

            x1 = np.clip(x1,0,img.shape[1]-1)
            x2 = np.clip(x2,0,img.shape[1]-1)

            y1 = np.clip(y1,0,img.shape[0]-1)
            y2 = np.clip(y2,0,img.shape[0]-1)

            img_ = img[y1:y2,x1:x2,:]

            #-----------------
            pts_tor_landmarks = []
            pts_hand = {}
            for ptk in range(21):
                xh,yh = pts[str(ptk)]["x"],pts[str(ptk)]["y"]
                xh = xh + bbox[0] -x1
                yh = yh + bbox[1] -y1
                pts_tor_landmarks.append([xh,yh])

                pts_hand[str(ptk)] = {
                    "x":xh,
                    "y":yh,
                    }
            #----------------
            if random.random() > 0.5: # 左右镜像
                img_ = cv2.flip(img_,1)
                pts_tor_landmarks = []
                pts_hand = {}
                for ptk in range(21):
                    xh,yh = pts[str(ptk)]["x"],pts[str(ptk)]["y"]
                    xh = xh + bbox[0] -x1
                    yh = yh + bbox[1] -y1
                    pts_tor_landmarks.append([img_.shape[1]-1-xh,yh])

                    pts_hand[str(ptk)] = {
                        "x":img_.shape[1]-1-xh,
                        "y":yh,
                        }

            pts_tor_landmarks_norm = []
            for i in range(len(pts_tor_landmarks)):
                x_ = float(pts_tor_landmarks[i][0])/float(abs(x2-x1))
                y_ = float(pts_tor_landmarks[i][1])/float(abs(y2-y1))
                pts_tor_landmarks_norm.append([x_,y_])
            #-----------------
            if self.vis:
                draw_bd_handpose(img_,pts_hand,0,0)

            img_ = cv2.resize(img_, self.img_size, interpolation = random.randint(0,5))

            if self.vis:
                cv2.namedWindow("hand_zfx",0)
                cv2.imshow("hand_zfx",img_)
                cv2.waitKey(1)
        #-------------------------------------
        if self.flag_agu == True:
            if random.random() > 0.5:
                c = float(random.randint(80,120))/100.
                b = random.randint(-10,10)
                img_ = contrast_img(img_, c, b)
        if self.flag_agu == True:
            if random.random() > 0.9:
                # print('agu hue ')
                img_hsv=cv2.cvtColor(img_,cv2.COLOR_BGR2HSV)
                hue_x = random.randint(-10,10)
                # print(cc)
                img_hsv[:,:,0]=(img_hsv[:,:,0]+hue_x)
                img_hsv[:,:,0] =np.maximum(img_hsv[:,:,0],0)
                img_hsv[:,:,0] =np.minimum(img_hsv[:,:,0],180)#范围 0 ~180
                img_=cv2.cvtColor(img_hsv,cv2.COLOR_HSV2BGR)
        if self.flag_agu == True:
            if random.random() > 0.95:
                img_ = img_agu_channel_same(img_)
        if self.vis == True:
            cv2.namedWindow('crop',0)
            cv2.imshow('crop',img_)
            cv2.waitKey(1)
        img_ = img_.astype(np.float32)
        img_ = (img_-128.)/256.
        img_ = img_.transpose(2, 0, 1)


        pts_tor_landmarks_norm = np.array(pts_tor_landmarks_norm).ravel()
        return img_,pts_tor_landmarks_norm
