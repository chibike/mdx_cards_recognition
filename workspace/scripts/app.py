#!/usr/bin/python3

r'''
----------------------------- App ----------------------------------
'''

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0,currentdir)
sys.path.insert(0,"{}/vector_illustration_processing".format(currentdir))

_BA_SUIT_ICONS_LOCATION = "{}/res/icons".format(currentdir)

import pi_point
import pi_line
import pi_path

from imutils import paths
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from matplotlib import pyplot as plt
import numpy as np
import random
import math
import cv2
import json
import time # used to calculate new image name

from prediction.guessing   import Guessing_Cards
from prediction.shallownet import ShallowNet_Cards
from prediction.lenet      import LeNet_Cards
from prediction.minivggnet import MiniVGGNet_Cards

class BaxterImageViewer(QWidget):
    def __init__(self, parent=None):
        super(BaxterImageViewer, self).__init__(parent)

        self.baxter_app = None

        self.camera_image = None
        self.focus_image = None

        self.closing = False

        self.__capture_device = None
        self.__capture_timer  = QTimer(self)
        self.__capture_timer.timeout.connect(self.handle_camera_update)
        self.__capture_timer.setInterval(100)

        self.init_UI()
        self.show()
    
    def closeEvent(self, event):
        self.closing = True

        if not self.baxter_app.closing:
            self.baxter_app.quit_app()

        self.disable_camera()
        event.accept()
    
    def keyPressEvent(self, event):
        key = event.key()

        if key in [Qt.Key_Q, Qt.Key_Escape]:
            self.quit_app()

    def quit_app(self):
        super(BaxterImageViewer, self).close()
    
    def init_UI(self):
        self.setWindowTitle("Image Viewer")
        self.setWindowModality(Qt.ApplicationModal)
        self.setFixedSize(525, 225)
        self.move(100, 400)

        self.camera_view_scene = QGraphicsScene()
        self.camera_view_view  = QGraphicsView(self.camera_view_scene, self)
        self.camera_view_view.resize(375, 225)
        self.camera_view_view.move(150, 0)

        self.focus_view_scene = QGraphicsScene()
        self.focus_view_view  = QGraphicsView(self.focus_view_scene, self)
        self.focus_view_view.resize(150, 225)
    
    def set_baxter_app(self, baxter_app):
        self.baxter_app = baxter_app
    
    def is_frame_valid(self, frame, nchannel=3):
        if frame is None:
            return False
        
        try:
            if len(frame.shape) != nchannel:
                return False
        except:
            return False
        
        return True
    
    def handle_camera_update(self):
        if self.__capture_device is None:
            return
        
        # TODO: handle error reading from device

        _, frame = self.__capture_device.read()

        if not self.is_frame_valid(frame):
            return
        
        # for cropped frame
        nh, nw = 320, 207; offset = 216
        
        self.camera_image = frame.copy()
        self.focus_image = frame[0:nh, offset:offset+nw, :]

        return self.render()
    
    def render(self):
        if not self.is_frame_valid(self.camera_image):
            return
        
        img = cv2.cvtColor(self.camera_image, cv2.COLOR_BGR2RGB)
        im_height, im_width, byte_val = img.shape

        ci_qimage = QImage(img.data, im_width, im_height, byte_val * im_width, QImage.Format_RGB888)
        ci_pix_map = QPixmap.fromImage(ci_qimage)

        self.camera_view_scene.clear()
        self.camera_view_scene.addPixmap(ci_pix_map)
        self.camera_view_view.fitInView(QRectF(0,0,im_width, im_height), Qt.KeepAspectRatio)
        self.camera_view_scene.update()

        
        
        
        if not self.is_frame_valid(self.focus_image):
            return
        
        img = cv2.cvtColor(self.focus_image, cv2.COLOR_BGR2RGB)
        im_height, im_width, byte_val = img.shape

        ci_qimage = QImage(img.data, im_width, im_height, byte_val * im_width, QImage.Format_RGB888)
        ci_pix_map = QPixmap.fromImage(ci_qimage)

        self.focus_view_scene.clear()
        self.focus_view_scene.addPixmap(ci_pix_map)
        self.focus_view_view.fitInView(QRectF(0,0,im_width, im_height), Qt.KeepAspectRatio)
        self.focus_view_scene.update()
    
    def enable_camera(self, camera_id):
        if self.__capture_device is not None:
            self.__capture_device.release()
        
        try:
            self.__capture_device = cv2.VideoCapture(camera_id)

            if not self.__capture_device.isOpened():
                raise IOError("Could not open image capture device with id {}".format(camera_id))
            
            self.__capture_timer.start()
        except:
            self.baxter_app.show_message(msg="Could not access your camera")
            self.__capture_device = None
    
    def disable_camera(self):
        self.__capture_timer.stop()
        if self.__capture_device is not None:
            self.__capture_device.release()

            del self.__capture_device
            self.__capture_device = None

    
class BaxterApp(QWidget):
    def __init__(self, parent=None):
        super(BaxterApp, self).__init__(parent)

        self.closing = False
        self.baxter_image_viewer = None

        __name_pre_formater = lambda n : "{}/{}_icons.png".format(_BA_SUIT_ICONS_LOCATION, n)
        self.icons = {
            "clubs"    : __name_pre_formater(  "clubs"),
            "spades"   : __name_pre_formater( "spades"),
            "hearts"   : __name_pre_formater( "hearts"),
            "diamonds" : __name_pre_formater("diamonds")
        }

        self.classifiers = {
            "ShallowNet" : ShallowNet_Cards,
            "LeNet"      : LeNet_Cards,
            "MiniVGGNet" : MiniVGGNet_Cards
        }

        self.camera_id = 0
        self.current_classifier = None
        self.selected_card_view = None
        
        self.init_UI()
        self.show()
    
    def closeEvent(self, event):
        reply = QMessageBox.question(self, "Quiting", "Do you want to exit?", QMessageBox.Yes, QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.closing = True

            if not self.baxter_image_viewer.closing:
                self.baxter_image_viewer.quit_app()

            event.accept()
        else:
            event.ignore()
    
    def keyPressEvent(self, event):
        key = event.key()

        if key in [Qt.Key_Q, Qt.Key_Escape]:
            self.quit_app()
        
    def init_UI(self):
        self.setWindowTitle("MDX Baxter Cards Selector")
        self.setWindowModality(Qt.ApplicationModal)
        self.setFixedSize(650, 445)

        x_padding = 10
        y_padding = 10
        card_view_width = 150
        card_view_height = 1.5 * card_view_width
        btn_height = 3 * y_padding

        section_heading = QLabel("Card Suits", self)
        section_heading.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        section_heading.setStyleSheet("QLabel {color: red; font: bold 14px;}")
        section_heading.move(x_padding, y_padding)

        self.card_01_scene  = QGraphicsScene()
        self.card_01_view = QGraphicsView(self.card_01_scene, self)
        self.card_01_view.resize(card_view_width, card_view_height)
        self.card_01_view.move(x_padding, (y_padding*3))

        self.card_02_scene  = QGraphicsScene()
        self.card_02_view = QGraphicsView(self.card_02_scene, self)
        self.card_02_view.resize(card_view_width, card_view_height)
        self.card_02_view.move((2 * x_padding) + card_view_width, (y_padding*3))

        self.card_03_scene  = QGraphicsScene()
        self.card_03_view = QGraphicsView(self.card_03_scene, self)
        self.card_03_view.resize(card_view_width, card_view_height)
        self.card_03_view.move((3 * x_padding)  + (2 * card_view_width), (y_padding*3))

        self.card_04_scene  = QGraphicsScene()
        self.card_04_view = QGraphicsView(self.card_04_scene, self)
        self.card_04_view.resize(card_view_width, card_view_height)
        self.card_04_view.move((4 * x_padding)  + (3 * card_view_width), (y_padding*3))


        
        self.card_01_identify_btn = QPushButton("Identify", self)
        self.card_01_identify_btn.setCheckable(False)
        self.card_01_identify_btn.resize(card_view_width, btn_height)
        self.card_01_identify_btn.move(x_padding, (y_padding*4) + card_view_height)
        # self.card_01_identify_btn.clicked.connect(self.card_01_identify_btn_callback)

        self.card_02_identify_btn = QPushButton("Identify", self)
        self.card_02_identify_btn.setCheckable(False)
        self.card_02_identify_btn.resize(card_view_width, btn_height)
        self.card_02_identify_btn.move((2 * x_padding) + card_view_width, (y_padding*4) + card_view_height)
        # self.card_02_identify_btn.clicked.connect(self.card_02_identify_btn_callback)

        self.card_03_identify_btn = QPushButton("Identify", self)
        self.card_03_identify_btn.setCheckable(False)
        self.card_03_identify_btn.resize(card_view_width, btn_height)
        self.card_03_identify_btn.move((3 * x_padding)  + (2 * card_view_width), (y_padding*4) + card_view_height)
        # self.card_03_identify_btn.clicked.connect(self.card_03_identify_btn_callback)

        self.card_04_identify_btn = QPushButton("Identify", self)
        self.card_04_identify_btn.setCheckable(False)
        self.card_04_identify_btn.resize(card_view_width, btn_height)
        self.card_04_identify_btn.move((4 * x_padding)  + (3 * card_view_width), (y_padding*4) + card_view_height)
        # self.card_04_identify_btn.clicked.connect(self.card_04_identify_btn_callback)

        
        
        self.card_01_select_btn = QPushButton("Select", self)
        self.card_01_select_btn.setCheckable(True)
        self.card_01_select_btn.resize(card_view_width, btn_height)
        self.card_01_select_btn.move(x_padding, (y_padding*5) + card_view_height + btn_height)
        self.card_01_select_btn.clicked.connect(self.card_01_select_btn_callback)

        self.card_02_select_btn = QPushButton("Select", self)
        self.card_02_select_btn.setCheckable(True)
        self.card_02_select_btn.resize(card_view_width, btn_height)
        self.card_02_select_btn.move((2 * x_padding) + card_view_width, (y_padding*5) + card_view_height + btn_height)
        self.card_02_select_btn.clicked.connect(self.card_02_select_btn_callback)

        self.card_03_select_btn = QPushButton("Select", self)
        self.card_03_select_btn.setCheckable(True)
        self.card_03_select_btn.resize(card_view_width, btn_height)
        self.card_03_select_btn.move((3 * x_padding)  + (2 * card_view_width), (y_padding*5) + card_view_height + btn_height)
        self.card_03_select_btn.clicked.connect(self.card_03_select_btn_callback)

        self.card_04_select_btn = QPushButton("Select", self)
        self.card_04_select_btn.setCheckable(True)
        self.card_04_select_btn.resize(card_view_width, btn_height)
        self.card_04_select_btn.move((4 * x_padding)  + (3 * card_view_width), (y_padding*5) + card_view_height + btn_height)
        self.card_04_select_btn.clicked.connect(self.card_04_select_btn_callback)

        
        section_heading = QLabel("Controls", self)
        section_heading.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        section_heading.setStyleSheet("QLabel {color: red; font: bold 14px;}")
        section_heading.move(x_padding, (y_padding*7) + card_view_height + (btn_height * 2))




        self.change_camera_btn = QPushButton("Change Camera", self)
        self.change_camera_btn.setCheckable(False)
        self.change_camera_btn.resize(card_view_width, (btn_height * 2))
        self.change_camera_btn.move(x_padding, (y_padding*9) + card_view_height + (btn_height * 2))
        self.change_camera_btn.clicked.connect(self.change_camera_btn_callback)

        self.select_classifier_cb = QComboBox(self)
        self.select_classifier_cb.addItems(list(self.classifiers.keys()))
        self.select_classifier_cb.resize(card_view_width, (btn_height * 2))
        self.select_classifier_cb.move((2 * x_padding) + card_view_width, (y_padding*9) + card_view_height + (btn_height * 2))
        # self.select_classifier_cb.currentIndexChanged.connect(self.select_classifier_cb_callback)

        self.detect_cards_btn = QPushButton("Detect Cards", self)
        self.detect_cards_btn.setCheckable(False)
        self.detect_cards_btn.resize(card_view_width, (btn_height * 2))
        self.detect_cards_btn.move((3 * x_padding)  + (2 * card_view_width), (y_padding*9) + card_view_height + (btn_height * 2))
        # self.detect_cards_btn.clicked.connect(self.detect_cards_btn_callback)

        self.fetch_card_btn = QPushButton("Fetch Card", self)
        self.fetch_card_btn.setCheckable(False)
        self.fetch_card_btn.resize(card_view_width, (btn_height * 2))
        self.fetch_card_btn.move((4 * x_padding)  + (3 * card_view_width), (y_padding*9) + card_view_height + (btn_height * 2))
        # self.fetch_card_btn.clicked.connect(self.fetch_card_btn_callback)

        self.w = QWidget(self)
        
        layout = QFormLayout()
        layout.setVerticalSpacing(10)
        layout.setHorizontalSpacing(10)

        section_heading = QLabel("Card Suits")
        section_heading.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        section_heading.setStyleSheet("QLabel {color: red; font: bold 14px;}")
        layout.addRow(section_heading)

        # self.w.setLayout(layout)
        # self.update_cards_view()
    
    def quit_app(self):
        super(BaxterApp, self).close()
    
    def set_baxter_image_viewer(self, baxter_image_viewer):
        self.baxter_image_viewer = baxter_image_viewer
    
    def show_message(self, title="Warning!", msg="Could not read image?"):
        QMessageBox.question(self, title, msg, QMessageBox.Ok, QMessageBox.Ok)
    
    def update_cards_view(self, cards_detected=("clubs", "spades", "hearts", "diamonds")):
        # for index, card_detected in enumerate(cards_detected):
        #     print ("[INFO] drawing {}".format(card_detected))
        #     if card_detected in self.icons.keys():
        #         icon = QPixmap(self.icons[card_detected])
                
        #         self.card_view_array[index].setPixmap(icon)
        #     else:
        #         print ("[ERROR] could not find {} in {}".format(card_detected, list(self.icons.keys())))
        pass
    
    def change_camera_btn_callback(self):
        cam_id, ok_pressed = QInputDialog.getInt(self, "Camera ID","Camera ID:", self.camera_id, 0, 50, 1)
        if not ok_pressed:
            return
        else:
            self.camera_id = cam_id
        
        self.baxter_image_viewer.enable_camera(self.camera_id)
    
    def card_01_select_btn_callback(self):
        if not self.card_01_select_btn.isChecked():
            return

        self.card_02_select_btn.setChecked(False)
        self.card_03_select_btn.setChecked(False)
        self.card_04_select_btn.setChecked(False)

        self.selected_card_view = 1
    
    def card_02_select_btn_callback(self):
        if not self.card_02_select_btn.isChecked():
            return

        self.card_01_select_btn.setChecked(False)
        self.card_03_select_btn.setChecked(False)
        self.card_04_select_btn.setChecked(False)

        self.selected_card_view = 2
    
    def card_03_select_btn_callback(self):
        if not self.card_03_select_btn.isChecked():
            return

        self.card_01_select_btn.setChecked(False)
        self.card_02_select_btn.setChecked(False)
        self.card_04_select_btn.setChecked(False)

        self.selected_card_view = 3
    
    def card_04_select_btn_callback(self):
        if not self.card_04_select_btn.isChecked():
            return

        self.card_01_select_btn.setChecked(False)
        self.card_02_select_btn.setChecked(False)
        self.card_03_select_btn.setChecked(False)

        self.selected_card_view = 4

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    biv = BaxterImageViewer()
    ba  = BaxterApp()

    biv.set_baxter_app(ba)
    ba.set_baxter_image_viewer(biv)

    sys.exit(app.exec_())
