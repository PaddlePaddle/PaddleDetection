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

from MyControl import *

dirname = os.path.dirname(PySide2.__file__)
plugin_path = os.path.join(dirname, 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path


class status():
    def __init__(self):
        # self.handleCalc()
        #正常情况下应该是先跳转到主页menu，现在主页还没完善好，因此先跳转到行人单目标检测
        self.handleCalc()

    def show_ui(self,location):
        loca="ui/"+location
        qfile_staus = QFile(loca)
        qfile_staus.open(QFile.ReadOnly)
        qfile_staus.close()
        self.ui = QUiLoader().load(qfile_staus)

    def help_set_shadow(self, x_offset
                        , y_offset
                        , radius
                        , color
                        , *control):
        for x in control:
            tempEffect = QGraphicsDropShadowEffect(self.ui)
            tempEffect.setOffset(x_offset, y_offset)
            tempEffect.setBlurRadius(radius)  # 阴影半径
            tempEffect.setColor(color)
            x.setGraphicsEffect(tempEffect)
    def help_set_style_sheet(self,s,*controllers):
        for x in controllers:
            x.setStyleSheet(s)

    def help_hide(self,*control):
        for x in control:
            x.setVisible(False)

    def help_set_up(self,*control):
        for x in control:
            x.raise_()

    def help_set_edit(self,edit,one):
        if one==1 or one==-1:
            self.time=self.time+one
            if self.time>999:self.time=999
            if self.time<0: self.time=0
            edit.setText(str(self.time))
        else:
            self.confi=int(self.confi*100+one*100)
            self.confi=float(self.confi/100.0)
            if self.confi > 1.0: self.confi = 1.0
            if self.confi < 0.0: self.confi = 0.0
            edit.setText(str(self.confi))

    def help_set_edit_by_hand(self,edit,one):
        if one==1:
            self.time=int(edit.text())
            now=int(edit.text())
            if self.time>999:self.time=999
            if self.time<0: self.time=0
            if now!=self.time:edit.setText(str(self.time))
        else:
            self.confi=int(float(edit.text())*100)
            self.confi = float(self.confi / 100.0)
            now=float(edit.text())
            if self.confi > 1.0: self.confi = 1.0
            if self.confi < 0.0: self.confi = 0.0
            if now!=self.confi:edit.setText(str(self.confi))

    def help_set_spinBox(self,edit,add,down,one):
        edit.textChanged.connect(lambda :self.help_set_edit_by_hand(edit,one))
        add.clicked.connect(lambda :self.help_set_edit(edit,one))
        down.clicked.connect(lambda :self.help_set_edit(edit,-1*one))

    def help_set_progress(self,len,progress,show_label):
        """
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        用来更新那个进度条的函数
        ！！！注意更新进度条的时候一定要先修改self.progressPos,
        self.progressPos是一个0到1的小数

        :param len:指的是总进度条的长度，是120
        :param progress: 指的是self.ui.label_progressBar,直接传这个参数就可以
        :param show_label:是self.ui.label_progressBar_num，直接传这个
        :return:
        """
        show_label.setText(str(int(self.progressPos*100))+"%")
        progress.resize(int(self.progressPos*len),progress.height())




    def handleCalc(self):
        self.show_ui("main_menu.ui")
        self.have_show_video = 0

        self.help_set_shadow(0, 0, 50, QColor(0, 0, 0, 0.06 * 255)
                             , self.ui.widget1
                             , self.ui.widget1_2
                             , self.ui.widget1_3)
        self.help_set_shadow(-10, 10, 30, QColor(0, 0, 0, 0.06 * 255)
                             , self.ui.widget
                             , self.ui.widget_3
                             , self.ui.widget_4
                             , self.ui.widget_5
                             , self.ui.widget_6
                             , self.ui.widget_7
                             , self.ui.widget_8)
        test_video=r"C:\Users\Administrator\Desktop\university\d4199e42f744cbe3be7f5ac262cd9056.mp4"
        label1=MyMenuVideoLabel("source/second/menu_car.PNG"
                                ,360,184,320,180
                                ,test_video
                                ,self.ui)
        label2=MyMenuVideoLabel("source/second/menu_pedestrian.PNG"
                                ,800,184,320,180
                                ,test_video
                                ,self.ui)

        label3 = MyMenuVideoLabel("source/second/menu_muti_object.PNG"
                                  , 1240, 184, 320, 180
                                  , test_video
                                  , self.ui)
        self.ui.show()





    def pedestrian_one_photo(self):
        self.show_ui("pedestrian_one_photo.ui")
        self.file_path = []
        # 先设置shadow
        self.help_set_shadow(0, 4, 0, QColor(221, 221, 221, 0.3 * 255)
                             , self.ui.label_3)
        self.help_set_shadow(0, 0, 50, QColor(221, 221, 221, 0.3 * 255)
                             , self.ui.widget_2, self.ui.widget_3)
        self.not_enter_ui = "pedestrian_one_photo_working.ui"
        self.is_enter_ui = "pedestrian_one_photo_working_enter.ui"
        self.confi = 0.0
        self.time = 0.0
        self.progressPos = 0.23
        self.ui.pushButton_addVideo.clicked.connect(
            lambda: self.load_video(1)
        )
        self.ui.show()

    def one_photo_change_enter(self):
        """
        每一次改变是否切换出入口，要先：
        1.配置基本需要用代码配置的ui
        2.手动调整的值要从原来的复制过去
        每个控件的值，除了手动调整的，其他都是根据推理结果更新，不用管
        3.将暂停键之类的各种键配置好
        :return:
        """
        result=[]
        result.append(self.lineEditConfi.text())
        result.append(self.ui.lineEdit_2.text())
        if self.is_enter==False:
            self.is_enter = True
            self.init_base_ui_for_one_photo_changing_enter(self.is_enter_ui)
        else:
            self.is_enter = False
            self.init_base_ui_for_one_photo_changing_enter(self.not_enter_ui)

        self.lineEditConfi.setText(result[0])
        self.ui.lineEdit_2.setText(result[1])
        self.load_video_controller()
        self.load_control_for_one_photo()
        return

    def init_base_ui_for_one_photo_changing_enter(self,ui_location):

        self.show_ui(ui_location)
        s = """QLineEdit{\nwidth: 80px;\nheight: 40px;\nbackground: #FFFFFF;\nborder-radius: 4px;\nborder: 1px solid #CCCCCC;\n\nfont-size: 18px;\nfont-family: AlibabaPuHuiTi_2_65_Medium;\ncolor: #333333;\nline-height: 26px;\nfont-weight:bold;\n}"""
        self.lineEditConfi = MyLineEdit(self.ui.label_confi_tip, s
                                        , 0, 352, 930, 57, 42, self.ui)
        s = """QPushButton{\nwidth: 126px;\nheight: 44px;\nbackground: #FFFFFF;\nborder-radius: 4px;\nborder: 1px solid #4E4EF2;\nfont-size: 18px;\nfont-family: AlibabaPuHuiTi_2_65_Medium;\ncolor: #4E4EF2;\nline-height: 26px;\nfont-weight:bold;\n}"""
        self.output_txt = MyPushButton(self.ui.label_txt_tip, s
                                       , "导出txt文件", 1419, 290, 126, 44, self.ui)
        self.ui.show()
        self.help_set_shadow(0, 4, 0, QColor(221, 221, 221, 0.3 * 255)
                             , self.ui.label_3)
        self.help_set_shadow(0, 0, 50, QColor(221, 221, 221, 0.3 * 255)
                             , self.ui.widget_2, self.ui.widget_3)

        self.ui.pushButton_10.clicked.connect(self.one_photo_change_enter)

    def load_control_for_one_photo(self):
        self.help_set_spinBox(self.lineEditConfi,self.ui.pushButton_7
                              ,self.ui.pushButton_11,0.01)
        self.help_set_spinBox(self.ui.lineEdit_2,self.ui.pushButton_5
                              ,self.ui.pushButton_6,1)

        self.help_set_progress(self.ui.widget_8.width(),self.ui.label_progressBar
                               ,self.ui.label_progressBar_num)
        return


    def load_video(self,video_count):
        """
        :param video_count: 要导入视频的数量，单镜头是1，跨境是2
        :return:
        """

        filePath=self.open_one_file_dialog("选择视频",0)
        if filePath=="":
            return
        self.file_path.append(filePath)
        if len(self.file_path)<video_count:
            return
        # 数量对不上就return，说明没有给够
        self.video_num=video_count
        self.is_enter=False
        # 一开始是没有打开
        self.init_base_ui_for_one_photo_changing_enter(self.not_enter_ui)
        if video_count==1:
            self.open_video()


    def open_video(self):
        self.frame1 = []
        self.cap1 = []
        self.timer_camera1 = []
        self.ui.label_7.setFixedSize\
            (self.ui.label_7.width(), self.ui.label_7.height())
        self.cap1 = cv2.VideoCapture(self.file_path[0])
        self.timer_camera1 = QTimer()
        self.load_video_controller()
        self.load_control_for_one_photo()

    def load_video_controller(self):
        if self.video_num==1:
            self.ui.pushButton.clicked.connect(self.video_start)
        self.ui.pushButton_2.clicked.connect(self.video_pause)
        self.ui.pushButton_3.clicked.connect(self.video_stop)

    def video_stop(self):
        self.timer_camera1.stop()
        self.cap1.release()
        # 可以让视频被清除掉，或者一些其他的功能

    def video_start(self):
         self.timer_camera1.start(100)
         self.timer_camera1.timeout.connect(self.OpenFrame1)

    def video_pause(self):
        self.timer_camera1.stop()

    def OpenFrame1(self):
        ret, frame = self.cap1.read()
        if ret:
            self.Display_Image(frame,self.ui.label_7)
        else:
            print("播放结束")
            self.cap1.release()
            self.timer_camera1.stop()

    def clear_video(self):
        if self.have_show_video==1:
            for i in range(len(self.timer_camera)):
                self.timer_camera[i].stop()
                self.cap[i].release()
            self.timer_camera=None
            self.cap=None
            self.have_show_video=0

    def Display_Image(self, image, controller):
        """
        ！！！！！！！！！！！！！！！！！！这里一定要看
        最关键的是这个函数
        参数image就是要展示在页面里的视频

        除此之外，对各种展示信息的更新，直接在这个函数里对相应控件进行更新就可以

        然后所有需要手动设置的参数，阙值和时间长度，只要在页面上设置好，这里就能知道
        阙值：self.confi
        时间: self.time
        :param image:
        :param controller:
        :return:
        """
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
        controller.setPixmap(QtGui.QPixmap(Q_img))
        controller.setScaledContents(True)

    def chose_all_self_picture(self):
        """
        全选
        :return:
        """
        count = self.ui.listWidget.count()
        for i in range(count):
            self.ui.listWidget.itemWidget(self.ui.listWidget.item(i))\
                .setChecked(True)

    def save_self_picture(self):
        """
        保存图片
        :return:
        """
        count = self.ui.listWidget.count()
        # 得到QListWidget的总个数
        cb_list = [self.ui.listWidget.itemWidget(self.ui.listWidget.item(i))
                   for i in range(count)]
        # 得到QListWidget里面所有QListWidgetItem中的QCheckBox
        # print(cb_list)
        chooses = []  # 存放被选择的数据
        for cb in cb_list:  # type:QCheckBox
            if cb.isChecked():
                chooses.append(cb.text())
        print(chooses)

    def clean_self_list_widget(self):
        self.ui.listWidget.clear()

    def switchType(self, type):
        """
        :param type: 1的话是图片 0的话是视频
        :return:
        """
        if type == 1:
            return "图片类型 (*.png *.jpg *.bmp)"
        else:
            return "视频类型 (*.mp3 *.mp4 *.flac)"

    """
    ------------------------------打开单个文件的函数
    """

    def open_one_file_dialog(self, title, type):
        """
        :param title: 标题
        :param type: 类型 1的话是图片 0的话是视频
        :return:
        """
        type = self.switchType(type)
        filePath, _ = QFileDialog.getOpenFileName(
            self.ui,  # 父窗口对象
            title,  # 标题
            r"C:\Users\Administrator\Desktop",  # 起始目录
            type  # 选择类型过滤项，过滤内容在括号中
        )
        return filePath

    def open_count_file_dialog(self, title, type, limit=-1):
        """
        :param title: 标题
        :param type: 类型
        :return:
        """
        type = self.switchType(type)
        filePath, _ = QFileDialog.getOpenFileNames(
            self.ui,  # 父窗口对象
            title,  # 标题
            r"C:\Users\Administrator\Desktop\university",  # 起始目录
            type  # 选择类型过滤项，过滤内容在括号中
        )
        if len(filePath) > limit and limit != -1:
            QMessageBox.critical(
                self.ui,
                '错误',
                '你选择的文件数量过多')

if __name__ == '__main__':

    app = QApplication([])
    MainWindow=QMainWindow()
    statu = status()
    statu.ui.show()
    app.exec_()



