#!/usr/bin/env python
r'''
----------------------------- Use App ----------------------------------
Implementation of The Cards Classification Code using a GUI

Please not that this script uses "Python 3" and the following additional libaries

    matplotlib, imutils, numpy, scipy, sklearn, keras, Pillow, and tensorflow

    Most of the other scripts have been written by "Chibuike Okpaluba", please read LICENSE.txt for more information.


DESCRIPTION

    This scripts launches a card dectection app which REQUIRES access to a CAMERA to work


HOW TO USE

    run: python use_app.py


FOR MORE INFORMATION

    Contact: co607@live.mdx.ac.uk
    Subject: MDX Cards Advanded Robotics Projects 2018

'''

from __future__ import division

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0,currentdir)
sys.path.insert(0,"{}/vector_illustration_processing".format(currentdir))

SAVE_IMG_PATH = "{}/../datasets/test_no_app".format(currentdir)

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
        self.btn_change_camera_id_callback()     # trigger camera selection dialog
        self.cb_classifier.setCurrentIndex(2)    # set the default prediction model to a LeNet based cnn
    
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

        section_heading = QLabel("Select Network Complexity")
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
        self.btn_predict_card.setCheckable(True)
        self.btn_predict_card.clicked.connect(self.btn_predict_card_callback)

        self.btn_change_camera_id = QPushButton("Change Camera")
        self.btn_change_camera_id.setCheckable(False)
        self.btn_change_camera_id.clicked.connect(self.btn_change_camera_id_callback)

        control_buttons = QHBoxLayout()
        control_buttons.addWidget(self.btn_predict_card)
        control_buttons.addWidget(self.btn_change_camera_id)

        section_heading = QLabel("Controls")
        section_heading.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        section_heading.setStyleSheet("QLabel {color: green; font: bold 14px;}")
        
        layout.addRow(section_heading)
        layout.addRow(self.btn_predict_card, self.btn_change_camera_id)

        self.w.setLayout(layout)
        self.w.move(70, 320)

        self.show()
    
    def quit_app(self):
        super(CardDetectionApp, self).close()

    def closeEvent(self, event):
        reply = QMessageBox.question(self, "Quiting", "Do you want to exit?", QMessageBox.Yes, QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.disable_camera_preview()
            
            event.accept()
        else:
            event.ignore()

    def keyPressEvent(self, event):
        key = event.key()

        if key == Qt.Key_S:
            if (self.image is None) or (len(self.image.shape) != 3):
                self.show_message(msg="Could not save empty image")
                return
            
            # create random name
            im_name = "{}{}{}.png".format(str(time.time()).split(".")[0][-5:], str(time.time()).split(".")[1], "".join([random.choice("abcdefghijklmnopqrstuvwxyz") for i in range(5)]))
            im_name = "{}/{}".format(SAVE_IMG_PATH, im_name)

            cv2.imwrite(im_name, self.image)
            self.show_message(title="Info", msg="Saved current image to {}".format(im_name))
        elif key in [Qt.Key_Q, Qt.Key_Escape]:
            self.quit_app()
        else:
            self.btn_predict_card_callback()

    def fim_render(self, image):
        # This method handles rendering the processed image

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
        # This method handles rendering the camera image

        if self.image is None: return
        if len(self.image.shape) != 3: return
        
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        # img = cv2.flip( img, 1 )                                         # Uncomment to flip image (mirror like view)
        im_height, im_width, byte_val = img.shape

        m_qimage = QImage(img.data, im_width, im_height, byte_val * im_width, QImage.Format_RGB888)
        m_pix_map = QPixmap.fromImage(m_qimage)

        self.im_graphics_scene.clear()
        self.im_graphics_scene.addPixmap(m_pix_map)
        self.im_graphics_view.fitInView(QRectF(0,0,im_width, im_height), Qt.KeepAspectRatio)
        self.im_graphics_scene.update()

        self.btn_predict_card_callback()
    
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
        if not self.btn_predict_card.isChecked(): return
        if self.image is None: return
        if len(self.image.shape) != 3: return
            
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
        # This method FILTERS and LABELS the given contours ----- THIS is the method that handles the card prediction
        
        def _get_coutour_image(img, path, contour):
            #  This function returns the subset (with padding) of the given image that contains the contour
            #    Two subsets are returned, (1. subset in original image), (2. subset in original image as a mask)

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
        
        def _constrain(x, mnx, mxx):
            return min(mxx, max(x, mnx))
        
        def determine_prediction(labels, probabilities):
            #  Given a random assortment of labels and probabilties, this function returns a list contain each label and its mean probability
            #  Don't try to comprehend how this works --> you can't :) (it's magic)

            f = lambda a, b : [list(filter(lambda x: x[0] == i, sorted(list(zip(a, b)), key=lambda x: x[0]))) for i in list(set(a))]

            def _get_prediction(foo, n):
                label, preds = list(zip(*foo))
                label = label[0]
                preds = list(sorted(preds)[-n:])
                return label, sum(preds) / float(len(preds))

            data = f(labels, probabilities)
            n_data = []

            for d in data:
                r = _get_prediction(d, 3)
                n_data.append((r[0], r[1]))
            
            # rearrange the list to ensure that label with the highest probability is first
            return list(sorted(n_data, key=lambda x : x[1], reverse=True))
        
        contours_list = [contour.reshape((contour.shape[0], 2)).tolist() for contour in contours]
        paths_list    = [pi_path.Path(raw_point_data=[pi_point.Point(x=point[0], y=point[1]) for point in contour_points], is_closed=True) for contour_points in contours_list]

        filtered_paths = []
        filtered_cnts = []
        labels = []
        probs = []


        # do not consider contours that have areas less than {@ min_allowed_area}
        min_allowed_area = 100
        paths_attributes  = [(path, path.rect_info.area, path.rect_info.perimeter) for path in paths_list if path.rect_info.area > min_allowed_area]
        paths_list, area_list, perimeter_list = zip(*paths_attributes)

        for path in paths_list:
            rect_info = path.rect_info

            # do not add contours with invalid an axes ratio, or area
            if path.ratio < 0.6: continue
            if rect_info.area < 800: continue
            if rect_info.area > 7000: continue
            
            # convert path to contour
            cnt = path.get_as_contour()

            # extract a padded section of image with the contour
            n_img, n_mask = _get_coutour_image(in_img, path, cnt)


            # THIS IS WHERE THE PREDICTION IS DONE
            if self.current_classifier is not None:
                label, prob = self.current_classifier.predict(n_img)
                
                if prob < 0.9: continue # Do not accept contours with probabilities < than 0.9

                labels.append(label)
                probs.append(prob)

            filtered_cnts.append(cnt)
            filtered_paths.append(path)
        
        preds = determine_prediction(labels, probs)
        number_labels = ["Ace", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten"]

        if len(preds) > 0:
            label, prob = preds[0] # select the prediction with the highest probability
            
            self.ed_card_suit.setText(label.capitalize())

            # set card number to a selection from {@ number_labels}, based on the number of filitered paths
            self.ed_card_number.setText(number_labels[_constrain(len(filtered_paths)-1, 0, len(number_labels)-1)])
        else:
            self.ed_card_suit.setText("None")
            self.ed_card_number.setText("None")
        
        cv2.drawContours(out_img, filtered_cnts, -1, (0,0,255), 1)
        return out_img, filtered_paths, filtered_cnts

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = CardDetectionApp()
    sys.exit(app.exec_())