# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 13:47:21 2023

@author: kesav
"""

import cv2
import numpy as np
'''
img=cv2.imread('C:/Users/kesav/Pictures/fruits.jpg')
cv2.imshow('frame',img)
key=cv2.waitKey(0) &v 0xFF

if key==ord('q'):
    cv2.destroyAllWindows()
    '''
    
cap=cv2.VideoCapture(1)
print(cap.isOpened)
while cap.isOpened():
    _,frame=cap.read()
    print(cap.get(cv2.CAP_PROF_FRAME_WIDTH))
    print(cap.get(cv2.CAP_PROF_FRAME_HEIGHT))
    cv2.imshow('Farme',frame)
    key=cv2.waitKey(1) & 0xFF
    if key==ord('q'):
        break
    
cv2.destroyAllWindows()
cap.release()