#!/usr/bin/env python
from __future__ import division

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0,currentdir)
sys.path.insert(0,"{}/vector_illustration_processing".format(currentdir))

import pi_point
import pi_line
import pi_path

import cv2
import json
from imutils import paths
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from matplotlib import pyplot as plt
import numpy as np
import math
import cv2
import random
import time

from prediction.guessing   import Guessing_Cards
from prediction.shallownet import ShallowNet_Cards
from prediction.lenet      import LeNet_Cards
from prediction.minivggnet import MiniVGGNet_Cards

class CardDetectionApp(QWidget):
    def __init__(self, parent=None):
        super(CardDetectionApp, self).__init__(parent)

        self.current_classifier = None
        self.image = None

        self.CAM_ID = 0
        self._capture_device = None

        self._cam_timer = QTimer(self)
        self._cam_timer.timeout.connect(self.handle_camera_refresh)
        self._cam_timer.setInterval(1000 / 10)
        
        self.init_UI()
    
    def init_UI(self):
        self.setWindowTitle("MDX Cards Detector")
        self.setWindowModality(Qt.ApplicationModal)
        self.setFixedSize(414, 550)

        self.im_graphics_scene = QGraphicsScene()
        self.im_graphics_view  = QGraphicsView(self.im_graphics_scene, self)
        self.im_graphics_view.resize(207, 320)

        self.fim_graphics_scene = QGraphicsScene()
        self.fim_graphics_view  = QGraphicsView(self.fim_graphics_scene, self)
        self.fim_graphics_view.resize(207, 320)
        self.fim_graphics_view.move(207, 0)

        self.w = QWidget(self)

        layout = QFormLayout()
        layout.setVerticalSpacing(10)
        layout.setHorizontalSpacing(10)

        self.cb_classifier = QComboBox()
        self.cb_classifier.addItems(["Guessing", "ShallowNet", "LeNet", "MiniVGGNet"])
        self.cb_classifier.currentIndexChanged.connect(self.cb_classifier_callback)

        self.ed_card_suit = QLineEdit()
        self.ed_card_suit.setReadOnly(True)
        
        self.ed_card_number = QLineEdit()
        self.ed_card_number.setReadOnly(True)

        section_heading = QLabel("Select Classifier")
        section_heading.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        section_heading.setStyleSheet("QLabel {color: red; font: bold 14px;}")
        layout.addRow(section_heading)
        layout.addRow(self.cb_classifier)

        section_heading = QLabel("Card Detected")
        section_heading.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        section_heading.setStyleSheet("QLabel {color: red; font: bold 14px;}")
        
        layout.addRow(section_heading)
        layout.addRow(QLabel("Suit"), self.ed_card_suit)
        layout.addRow(QLabel("#"), self.ed_card_number)

        self.btn_predict_card = QPushButton("Predict")
        self.btn_predict_card.setCheckable(False)
        self.btn_predict_card.clicked.connect(self.btn_predict_card_callback)

        self.btn_change_camera_id = QPushButton("Change Camera")
        self.btn_change_camera_id.setCheckable(False)
        self.btn_change_camera_id.clicked.connect(self.btn_change_camera_id_callback)

        control_buttons = QHBoxLayout()
        control_buttons.addWidget(self.btn_predict_card)
        control_buttons.addWidget(self.btn_change_camera_id)

        section_heading = QLabel("Card #")
        section_heading.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        section_heading.setStyleSheet("QLabel {color: red; font: bold 14px;}")
        
        layout.addRow(section_heading)
        layout.addRow(self.btn_predict_card, self.btn_change_camera_id)

        self.w.setLayout(layout)
        self.w.move(70, 320)

        self.show()
    
    def closeEvent(self, event):
        self.disable_camera_preview()
        event.accept()
    
    def fim_render(self, image):
        if image is None: return
        if len(image.shape) != 3: return
        
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im_height, im_width, byte_val = img.shape

        m_qimage = QImage(img.data, im_width, im_height, byte_val * im_width, QImage.Format_RGB888)
        m_pix_map = QPixmap.fromImage(m_qimage)

        self.fim_graphics_scene.clear()
        self.fim_graphics_scene.addPixmap(m_pix_map)
        self.fim_graphics_view.fitInView(QRectF(0,0,im_width, im_height), Qt.KeepAspectRatio)
        self.fim_graphics_scene.update()
    
    def im_render(self):
        if self.image is None: return
        if len(self.image.shape) != 3: return
        
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        im_height, im_width, byte_val = img.shape

        m_qimage = QImage(img.data, im_width, im_height, byte_val * im_width, QImage.Format_RGB888)
        m_pix_map = QPixmap.fromImage(m_qimage)

        self.im_graphics_scene.clear()
        self.im_graphics_scene.addPixmap(m_pix_map)
        self.im_graphics_view.fitInView(QRectF(0,0,im_width, im_height), Qt.KeepAspectRatio)
        self.im_graphics_scene.update()
    
    def show_message(self, title="Warning!", msg="Could not read image?"):
        QMessageBox.question(self, title, msg, QMessageBox.Ok, QMessageBox.Ok)
    
    def handle_camera_refresh(self):
        if self._capture_device is None: return

        _, frame = self._capture_device.read()

        if frame is None: return
        if len(frame.shape) != 3: return
        
        # crop frame
        nh, nw = 320, 207; offset = 216
        frame = frame[0:nh,offset:offset+nw,:]

        self.image = frame.copy()
        self.im_render()
    
    def enable_camera_preview(self):
        if self._capture_device is not None:
            self._capture_device.release()
        
        try:
            self._capture_device = cv2.VideoCapture(self.CAM_ID)
            if not self._capture_device.isOpened():
                raise IOError("Could not open image capture device with id {}".format(self.CAM_ID))

            self._cam_timer.start()
        except:
            self.show_message(msg="Could not access your camera")
            self._capture_device = None
        
    def disable_camera_preview(self):
        self._cam_timer.stop()
        if self._capture_device is not None:
            self._capture_device.release()

            del self._capture_device
            self._capture_device = None
    
    def btn_change_camera_id_callback(self):
        cam_id, ok_pressed = QInputDialog.getInt(self, "Camera ID","Camera ID:", self.CAM_ID, 0, 50, 1)
        if not ok_pressed:
            return
        else:
            self.CAM_ID = cam_id
            self.enable_camera_preview()
    
    def btn_predict_card_callback(self):
        frame = self.image.copy()
        processed_frame = frame.copy()

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

        new_frame = gray_frame.copy()
        new_frame[:,:] = 0

        gray_mean  = int(np.mean(gray_frame.ravel()))
        ret, gray_th  = cv2.threshold(gray_frame  , gray_mean , 255, cv2.THRESH_BINARY)

        kernel = np.ones((3,3),np.uint8)
        gray_th = cv2.erode(gray_th, kernel, iterations = 1)

        _, contours, hierarchy = cv2.findContours(gray_th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        processed_frame, _, _ = self.process_contours(contours, frame, processed_frame)
        self.fim_render(processed_frame)
    
    def cb_classifier_callback(self, index):
        text = self.cb_classifier.currentText()
        
        if text == "ShallowNet":
            self.current_classifier = ShallowNet_Cards.SuitDetector()
        elif text == "LeNet":
            self.current_classifier = LeNet_Cards.SuitDetector()
        elif text == "Guessing":
            self.current_classifier = Guessing_Cards.SuitDetector()
        elif text == "MiniVGGNet":
            self.current_classifier = MiniVGGNet_Cards.SuitDetector()
    
    def process_contours(self, contours, in_img, out_img):
        
        def _get_coutour_image(img, path, contour):
            width = path.rect_info.width
            height = path.rect_info.height

            im_h, im_w = img.shape[:2]
            mask = np.zeros((im_h, im_w))
            cv2.drawContours(mask, [contour], -1, 255, -1)

            padding = 10
            top_x = max(path.rect_info.top_left.x - padding, 0)
            bottom_x = min(path.rect_info.bottom_right.x + padding, im_w)

            top_y = max(path.rect_info.top_left.y - padding, 0)
            bottom_y = min(path.rect_info.bottom_right.y + padding, im_h)

            n_mask = mask[top_y:bottom_y, top_x:bottom_x]
            n_img  = img[top_y:bottom_y, top_x:bottom_x]
            
            return n_img, n_mask

        contours_list = [contour.reshape((contour.shape[0], 2)).tolist() for contour in contours]
        paths_list    = [pi_path.Path(raw_point_data=[pi_point.Point(x=point[0], y=point[1]) for point in contour_points], is_closed=True) for contour_points in contours_list]

        filtered_paths = []
        filtered_cnts = []
        labels = []
        preds = []

        min_allowed_area = 100
        paths_attributes  = [(path, path.rect_info.area, path.rect_info.perimeter) for path in paths_list if path.rect_info.area > min_allowed_area]
        paths_list, area_list, perimeter_list = zip(*paths_attributes)

        for path in paths_list:
            rect_info = path.rect_info
            cnt = path.get_as_contour()

            n_img, n_mask = _get_coutour_image(in_img, path, cnt)
            if self.current_classifier is not None:
                label, pred = self.current_classifier.predict(n_img)
                if pred < 0.9: continue

                print ("[INFO] prediction: {}".format((label, pred)))

            filtered_cnts.append(cnt)
            filtered_paths.append(path)

        
        cv2.drawContours(out_img, filtered_cnts, -1, (0,0,255), 1)
        return out_img, filtered_paths, filtered_cnts
    
    def label_contours(self, contours):
        pass


if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = CardDetectionApp()
    sys.exit(app.exec_())