#!/usr/bin/env python
from __future__ import division

import os,sys,inspect
sys.path.insert(0,"D:\\Chibuike\\Projects\\Myself\\Python\\Python27\\vector_illustration_processing")
# sys.path.insert(0,"/mnt/d/Chibuike/Projects/Myself/Python/Python27/vector_illustration_processing")

from matplotlib import pyplot as plt
from PIL import Image
import pytesseract
import numpy as np
import math
import cv2
import random
import time

import pi_point
import pi_line
import pi_path

font = cv2.FONT_HERSHEY_SIMPLEX

class_name = "three_sisters"

def save_image(img, path, contour):
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

    im_name = "{}{}{}.png".format(str(time.time()).split(".")[0][-5:], str(time.time()).split(".")[1], "".join([random.choice("abcdefghijklmnopqrstuvwxyz") for i in range(5)]))
    cv2.imwrite("images/{}/img/{}".format(class_name, im_name), n_img)
    cv2.imwrite("images/{}/mask/{}".format(class_name, im_name), n_mask)

    print "Saved"


def get_contour_color_from_image(contour, img):
    mask = np.zeros(img.shape[:2])
    cv2.drawContours(mask, [contour], -1, 255, -1)

    color_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[np.where(mask > 127)]

    h_mean = np.mean(color_frame[:,0])
    s_mean = np.mean(color_frame[:,1])
    v_mean = np.mean(color_frame[:,2])

    return (h_mean, s_mean, v_mean)
    # return (0, 0, 0)


def approx_contour(contour, resolution=0.001):
    epsilon = resolution*cv2.arcLength(contour,True)
    return cv2.approxPolyDP(contour,epsilon,True)

def process_contours(contours, in_img, out_img):
    # contours = [approx_contour(contour) for contour in contours]
    contours_list = [contour.reshape((contour.shape[0], 2)).tolist() for contour in contours]
    paths_list    = [pi_path.Path(raw_point_data=[pi_point.Point(x=point[0], y=point[1]) for point in contour_points], is_closed=True) for contour_points in contours_list]

    filtered_paths = []
    c_h = []
    c_s = []
    c_v = []

    min_allowed_area = 100
    paths_attributes  = [(path, path.rect_info.area, path.rect_info.perimeter) for path in paths_list if path.rect_info.area > min_allowed_area]
    paths_list, area_list, perimeter_list = zip(*paths_attributes)

    area_list = list(area_list)
    perimeter_list = list(perimeter_list)

    if len(paths_list) > 2:
        area_std = np.std(area_list)
        perimeter_std = np.std(perimeter_list)

        print "area_std: ", area_std, " vs perimeter_std: ", perimeter_std, " # ", len(paths_list)
        
        area_list.remove(max(area_list))
        area_list.remove(min(area_list))

    area_list = np.array(area_list)
    perimeter_list = np.array(perimeter_list)

    area_mean = np.mean(area_list)
    perimeter_mean = np.mean(perimeter_list)

    th = 0.95
    area_max = area_mean + (th * area_mean)
    area_min = area_mean - (th * area_mean)

    perimeter_max = perimeter_mean + (th * perimeter_mean)
    perimeter_min = perimeter_mean - (th * perimeter_mean)

    # print "-"*10

    for path in paths_list:
        rect_info = path.rect_info

        # if path.ratio < 0.6: continue
        if (rect_info.area < area_min or rect_info.area > area_max):
            continue

        if (rect_info.perimeter < perimeter_min or rect_info.perimeter > perimeter_max) < 0:
            continue

        cnt = path.get_as_contour()
        h,s,v = get_contour_color_from_image(cnt, in_img)

        filtered_paths.append(path)
        c_h.append(h); c_s.append(s); c_v.append(v)

        print ("[INFO] ({}, {}, {})".format(len(filtered_paths), h, s, v))

        cv2.drawContours(out_img, [cnt], -1, (0,0,255), 1)
        # print path.count_edges(math.radians(100))
        # print ratio, rect_info.area, rect_info.perimeter, cv2.moments(cnt), cv2.isContourConvex(cnt)

        save_image(in_img, path, cnt)
    
    # text = ""
    
    # if len(c_h) > 0:
    #     text = "#{} ".format(len(filtered_paths))
    #     if np.mean(c_s) < 100:
    #         text += "SPADES or THREE-SISTERS"
    #         # print (text, "c ({}, {}, {})".format(len(filtered_paths), np.mean(c_h), np.mean(c_s), np.mean(c_v)))
    #     else:
    #         text += "HEART or DIAMOND"
    #         # print (text, "c ({}, {}, {})".format(len(filtered_paths), np.mean(c_h), np.mean(c_s), np.mean(c_v)))

    
    # cv2.putText(out_img,text,(0,20), font, 0.4,(255,255,255),1,cv2.LINE_AA)

    return filtered_paths


def draw_paths(image, paths):
    for path in paths:
        cv2.drawContours(image, [path.get_as_contour()], -1, (0,0,255), 1)
    return image

cap = cv2.VideoCapture(0)

while (True):
    ret, frame = cap.read()

    ih, iw, _ = frame.shape
    nh = int(ih / 1.5)

    nw = (nh * 5.5) / 8.5
    offset = int((iw - nw) / 2.0)
    nw = int(nw)

    frame = frame[0:nh,offset:offset+nw,:]
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
    # cv2.drawContours(frame, contours, -1, (0,255,0), 3)

    cv2.imshow('frame', frame)
    # cv2.imshow('gray_th', gray_th)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('h'):
        plt.hist(new_frame.ravel(),256,[0,256]); plt.show()
    elif key == ord('a'):
        p = process_contours(contours, frame, processed_frame)
        # processed_frame = draw_paths(processed_frame, p)
        cv2.imshow('pframe', processed_frame)


cap.release()
cv2.destroyAllWindows()