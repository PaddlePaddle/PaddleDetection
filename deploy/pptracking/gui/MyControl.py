from PySide2.QtWidgets import  *
from PySide2.QtUiTools import *
from PySide2.QtCore import *
from PySide2 import QtGui,QtCore,QtWidgets
from PySide2.QtGui import *

import cv2
import sys, os
import PySide2
import firstSource
import secondSource
class MyPushButton(QPushButton):
    def __init__(self,control,style,text,x,y,w,h,parent=None):
        QPushButton.__init__(self,parent)
        # control表示的是这个控件的提示信息控件
        self.control=control
        self.control.setVisible(False)
        self.setStyleSheet(style)
        self.setText(text)
        self.move(x,y)
        self.resize(w,h)
        self.raise_()

    def enterEvent(self, event):
        self.control.setVisible(True)

    def leaveEvent(self, event):
        self.control.setVisible(False)

class MyLineEdit(QLineEdit):
    def __init__(self,control,style,text,x,y,w,h,parent=None):
        QLineEdit.__init__(self,parent)
        # control表示的是这个控件的提示信息控件
        self.control=control
        self.control.setVisible(False)
        self.setStyleSheet(style)
        text = str(text)
        self.setText(text)
        self.move(x,y)
        self.resize(w,h)
        self.raise_()

    def enterEvent(self, event):
        self.control.setVisible(True)

    def leaveEvent(self, event):
        self.control.setVisible(False)


class MyMenuVideoControl(QLabel):
    def mouseMoveEvent(self, event) :
        self.setVisible(True)
    def mousePressEvent(self, QMouseEvent):
        if QMouseEvent.buttons() == QtCore.Qt.LeftButton:
            if self.start == False:
                self.setPixmap(QtGui.QPixmap(self.start_image))
                self.timer_camera.start(50)
                self.timer_camera.timeout.connect(self.OpenFrame1)
                self.start = True
            else:
                self.setPixmap(QtGui.QPixmap(self.pause_image))
                self.start = False
                self.timer_camera.stop()

    def OpenFrame1(self):
        ret, frame = self.cap.read()
        if ret:
            self.Display_Image(frame)
        else:
            self.cap.release()
            self.timer_camera.stop()

    def Display_Image(self, image):
        if (len(image.shape) == 3):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            Q_img = QImage(image.data,
                           image.shape[1],
                           image.shape[0],
                           QImage.Format_RGB888)
        elif (len(image.shape) == 1):
            Q_img = QImage(image.data,
                           image.shape[1],
                           image.shape[0],
                           QImage.Format_Indexed8)
        else:
            Q_img = QImage(image.data,
                           image.shape[1],
                           image.shape[0],
                           QImage.Format_RGB888)
        self.show_lable.setPixmap(QtGui.QPixmap(Q_img))
        self.show_lable.setScaledContents(True)
    # 这是那个播放键和暂停键
    def __init__(self,x,y,w,h,show_label,video,parent):
        QLabel.__init__(self,parent)
        self.pause_image="source/second/pause.png"
        self.start_image="source/second/start.png"
        self.start=False
        self.setPixmap(QtGui.QPixmap(self.pause_image))
        self.setScaledContents(True)
        self.cap = []
        self.timer_camera = QTimer()
        self.cap = cv2.VideoCapture(video)
        self.show_lable=show_label
        self.move(x,y)
        self.resize(w,h)

class MyMenuVideoLabel(QLabel):
    def __init__(self,picture,x,y,w,h,video,parent=None):
        # 这是一个播放展示视频的控件，control是那个控制它的元件
        QLabel.__init__(self,parent)
        # control表示的是这个控件的提示信息控件
        self.setPixmap(QtGui.QPixmap(picture))
        self.setScaledContents(True)
        self.move(x,y)
        self.resize(w,h)
        self.raise_()
        self.setFixedSize(self.width(),self.height())
        self.control=MyMenuVideoControl(x+140,y+70,40,40,self,video,parent)

    def enterEvent(self, event):
        self.control.setVisible(True)

    def leaveEvent(self, event):
        self.control.setVisible(False)