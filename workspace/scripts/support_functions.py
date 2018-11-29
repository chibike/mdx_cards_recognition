#!/usr/bin/env python
r'''
----------------------------- Support Functions ----------------------------------
Contains support functions used by other scripts

Please not that this script uses "Python 3" and the following additional libaries

    matplotlib, imutils, numpy, scipy, sklearn, keras, Pillow, and tensorflow

    Most of the other scripts have been written by "Chibuike Okpaluba", please read LICENSE.txt for more information.

    Most importantly, the contents the "vector_illustration_processing" folder MUST NOT be distributed beyond the staff and students at Middlesex University as it contains some
    properitory code that is still being developed.

    Thank you for understanding :)

FOR MORE INFORMATION

    Contact: co607@live.mdx.ac.uk
    Subject: MDX Cards Advanded Robotics Projects 2018

'''


from __future__ import division

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0,currentdir)
sys.path.insert(0,"{}/vector_illustration_processing".format(currentdir))

import pi_point
import pi_line
import pi_path

from matplotlib import pyplot as plt
from imutils import paths
import numpy as np
import random
import json
import math
import time
import cv2

from prediction.guessing   import Guessing_Cards
from prediction.shallownet import ShallowNet_Cards
from prediction.lenet      import LeNet_Cards
from prediction.minivggnet import MiniVGGNet_Cards

from support_functions import *

def process_contours(contours, in_img, out_img, prediction_model=None):
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

        in_img_w, in_img_h,_ = (0, 0, 0)

        if len(in_img.shape) == 3:
            in_img_w, in_img_h,_ = in_img.shape
        else:
            raise ValueError("The input image must be of type cv2::mat bgr")

        contours_list = [contour.reshape((contour.shape[0], 2)).tolist() for contour in contours]
        paths_list    = [pi_path.Path(raw_point_data=[pi_point.Point(x=point[0], y=point[1]) for point in contour_points], is_closed=True) for contour_points in contours_list]

        filtered_paths = []
        filtered_cnts = []
        labels = []
        probs = []

        min_allowed_area = (in_img_w * in_img_h) * (500.0 / 66240.0)
        paths_attributes  = [(path, path.rect_info.area, path.rect_info.perimeter) for path in paths_list if path.rect_info.area > min_allowed_area]
        paths_list, area_list, perimeter_list = zip(*paths_attributes)

        for path in paths_list:
            rect_info = path.rect_info

            if path.ratio < 0.6: continue
            if rect_info.area > (in_img_w * in_img_h) * (7000.0 / 66240.0): continue
            
            cnt = path.get_as_contour()

            n_img, n_mask = _get_coutour_image(in_img, path, cnt)
            if prediction_model is not None:
                label, prob = prediction_model.predict(n_img)
                if prob < 0.9: continue

                labels.append(label)
                probs.append(prob)

            filtered_cnts.append(cnt)
            filtered_paths.append(path)
        
        def _constrain(x, mnx, mxx):
            return min(mxx, max(x, mnx))
        
        def determine_prediction(labels, probabilities):
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

            return list(sorted(n_data, key=lambda x : x[1], reverse=True))
        
        preds = determine_prediction(labels, probs)
        number_labels = ["Ace", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten"]

        label = "None"
        number = "None"
        probability = 1.0

        if len(preds) > 0:
            label, probability = preds[0]
            label = label.capitalize()
            number = number_labels[_constrain(len(filtered_paths)-1, 0, len(number_labels)-1)]
        
        cv2.drawContours(out_img, filtered_cnts, -1, (0,255,0), 1)
        return out_img, [label, number, probability]

def process_image(image_path, prediction_model=None):
    frame = cv2.imread(image_path)
    processed_frame = frame.copy()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    gray_mean  = int(np.mean(gray_frame.ravel()))
    ret, gray_th  = cv2.threshold(gray_frame, gray_mean, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3,3),np.uint8)
    gray_th = cv2.erode(gray_th, kernel, iterations=1)

    _, contours, _ = cv2.findContours(gray_th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return process_contours(contours, frame, processed_frame, prediction_model), gray_th

def combine_images(img1, img2, pad_width=10):
    if img1.shape != img2.shape:
        raise ValueError("The given images must have similar shapes")

    im_h, im_w, _ = img1.shape

    pad_width = int(pad_width)
    n_im_w = int((im_w * 2) + pad_width)

    n_img = np.zeros((im_h, n_im_w, 3), dtype=np.uint8)

    n_img[:, :im_w, :] = img1
    n_img[:, (im_w + pad_width):, :] = img2

    return n_img
