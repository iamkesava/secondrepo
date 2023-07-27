# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 20:30:44 2023

@author: kesav
"""

import cv2
import imutils
import time
from PIL import Image

'''
vr=cv2.VideoCapture(1)
time.sleep(1)
firstFrame=None
area=500

while True:
    _,img=vr.read()
    text='Normal'
    img=imutils.resize(img,width=500)
    grayimg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gaussinimg=cv2.GaussianBlur(grayimg,(21,21),0) 
    if firstFrame is None:
        firstFrame=gaussinimg
        continue
    imgdiff=cv2.absdiff(firstFrame,gaussinimg)
    threshimg=cv2.threshold(imgdiff,25,255,cv2.THRESH_BINARY)[1]
    thresimg=cv2.dilate(threshimg,None,iterations=2)
    cnts=cv2.findContours(threshimg.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts=imutils.grab_contours(cnts)
    for c in cnts:
        if cv2.contourArea(c)<area:
            continue
        (x,y,w,h)=cv2.boundingRect(c)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        text='moving object detected'
    print(text)
    cv2.putText(img,text,(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
    cv2.imshow('frame',img)
    key=cv2.waitKey(1)& 0xFF
    if key==ord('q'):
        break

vr.release()
cv2.destroyAllWindows()'''


vr=cv2.VideoCapture(1)
time.sleep(1)
firstFrame=None
area=500

while True:
    _,img=vr.read()
    text='Normal'
    img=imutils.resize(img,width=500)
    grayimg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gaussinimg=cv2.GaussianBlur(grayimg,(21,21),0) 
    if firstFrame is None:
        firstFrame=gaussinimg
        continue
    imgdiff=cv2.absdiff(firstFrame,gaussinimg)
    cv2.imshow('frame',imgdiff)
    key=cv2.waitKey(1)& 0xFF
    if key==ord('q'):
        break

vr.release()
cv2.destroyAllWindows()
'''

a=cv2.imread('C:/Users/kesav/Pictures/numberplate.webp',cv2.IMREAD_GRAYSCALE)
b=cv2.imread('C:/Users/kesav/Pictures/numberplate.webp',cv2.IMREAD_GRAYSCALE)
c=cv2.absdiff(a,b)
cv2.imshow('a',c)
cv2.waitKey(0)
'''