#!/usr/bin/python3

import numpy as np
import cv2, PIL
import cv2.aruco as aruco
import matplotlib.pyplot as plt
import matplotlib as mpl

cap = cv2.VideoCapture(0)

while (True):
    ret, frame = cap.read()

    # preprocessing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
    params = aruco.DetectorParameters_create()
    corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict, parameters=params)

    frame = aruco.drawDetectedMarkers(frame.copy(), corners, ids)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF

    if key in [ord('q'), 27]:
        break

cap.release()
cv2.destroyAllWindows()

