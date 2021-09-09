#-*-coding:utf-8-*-
# date:2021-12-05
# Author: Eric.Lee
## function: data agu

import numpy as np
import cv2

#-------------------------------------------------------------------------------
# eye_left_n,eye_right_n:为扰动后的参考点坐标

def hand_alignment_aug_fun(imgn,eye_left_n,eye_right_n,\
facial_landmarks_n = None,\
angle = None,desiredLeftEye=(0.34, 0.42),desiredFaceWidth=160, desiredFaceHeight=None,draw_flag = False):

    if desiredFaceHeight is None:
        desiredFaceHeight = desiredFaceWidth

    leftEyeCenter = eye_left_n
    rightEyeCenter = eye_right_n
    # compute the angle between the eye centroids
    dY = rightEyeCenter[1] - leftEyeCenter[1]
    dX = rightEyeCenter[0] - leftEyeCenter[0]
    if angle == None:
        angle = np.degrees(np.arctan2(dY, dX))
    else:
        # print('  a) disturb angle : ',angle)
        angle += np.degrees(np.arctan2(dY, dX))#基于正对角度的扰动
        # print('  b) disturb angle : ',angle)

    # compute the desired right eye x-coordinate based on the
    # desired x-coordinate of the left eye
    desiredRightEyeX = 1.0 - desiredLeftEye[0]
	# determine the scale of the new resulting image by taking
	# the ratio of the distance between eyes in the *current*
	# image to the ratio of distance between eyes in the
	# *desired* image
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desiredDist = (desiredRightEyeX - desiredLeftEye[0])
    desiredDist *= desiredFaceWidth
    scale = desiredDist / dist
    # compute center (x, y)-coordinates (i.e., the median point)
    # between the two eyes in the input image
    eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) / 2,(leftEyeCenter[1] + rightEyeCenter[1]) / 2)
    # grab the rotation matrix for rotating and scaling the face
    M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
    # update the translation component of the matrix
    tX = desiredFaceWidth * 0.5
    tY = desiredFaceHeight * desiredLeftEye[1]
    M[0, 2] += (tX - eyesCenter[0])
    M[1, 2] += (tY - eyesCenter[1])

    M_reg = np.zeros((3,3),dtype = np.float32)
    M_reg[0,:] = M[0,:]
    M_reg[1,:] = M[1,:]
    M_reg[2,:] = (0,0,1.)
    # print(M_reg)
    M_I = np.linalg.inv(M_reg)#矩阵求逆，从而获得，目标图到原图的关系
    # print(M_I)
    # apply the affine transformation
    (w, h) = (desiredFaceWidth, desiredFaceHeight)
    output = cv2.warpAffine(imgn, M, (w, h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)# INTER_LINEAR INTER_CUBIC INTER_NEAREST
    #BORDER_REFLECT BORDER_TRANSPARENT BORDER_REPLICATE CV_BORDER_WRAP BORDER_CONSTANT

    pts_landmarks = []

    for k in range(len(facial_landmarks_n)):
        x = facial_landmarks_n[k][0]
        y = facial_landmarks_n[k][1]

        x_r = (x*M[0][0] + y*M[0][1] + M[0][2])
        y_r = (x*M[1][0] + y*M[1][1] + M[1][2])
        pts_landmarks.append([x_r,y_r])
        # if draw_flag:
        #     cv2.circle(output, (int(x_r),int(y_r)), np.int(1),(0,0,255), 1)


        #
        # cv2.circle(output, (ptx2,pty2), np.int(1),(0,0,255), 1)
        # cv2.circle(output, (ptx3,pty3), np.int(1),(0,255,0), 1)




    return output,pts_landmarks,M_I
