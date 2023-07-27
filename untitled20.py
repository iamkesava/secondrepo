# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 23:10:50 2023

@author: kesav
"""
'''
import cv2
from cvzone.PoseModule import PoseDetector
import numpy as np

detector = PoseDetector() 
#img=cv2.imread('C:/Users/kesav/Pictures/OIP.jpg')

cap=cv2.VideoCapture(1)
while cap.isOpened():
    _,img=cap.read()
    img = detector.findPose(img)
    imlist, bbox = detector.findPosition(img,bboxWithHands=True)
    measure_list=[]
    b=bbox.items()
    
    for i in b:
        if len(i[1])==2:
            x=i[1][0]
            y=i[1][0]
            cv2.circle(img,i[1],10,(0,0,255),-1)
            cv2.putText(img,'width {}'.format(round(measure_list[2],1)),(x-50,y-15),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
            cv2.putText(img,'height {}'.format(round(measure_list[3],1)),(x-50,y+15),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
            print(i[1])
            
        elif len(i[1])>2:
            for n in i[1]:
                measure_list.append(n)
                #print(n)
            print(measure_list)
            cv2.rectangle(img,(measure_list[0],measure_list[1],measure_list[2],measure_list[3]),(0,255,0),10)
        
    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
    
cv2.destroyAllWindows()
cap.release()
'''

'''
detector = PoseDetector()

cap=cv2.VideoCapture(1)
while cap.isOpened():
    _,img=cap.read()
    img = detector.findPose(img)
    imlist, bbox = detector.findPosition(img,bboxWithHands=True)
    if bbox:
        imlist=bbox['bbox']
        print(imlist)
        x1,y1=imlist[5]
        
    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
    
cv2.destroyAllWindows()
cap.release()'''
'''
import cv2
from cvzone.PoseModule import PoseDetector

detector = PoseDetector()

cap=cv2.VideoCapture(1)
while cap.isOpened():
    _,img=cap.read()
    img = detector.findPose(img)
    imlist, bbox = detector.findPosition(img,bboxWithHands=True)
    #print(imlist)

    #if imlist:
        
    cv2.imshow('frame',img)
    if cv2.waitKey(1)==ord('q'):
        break
       
    
cv2.destroyAllWindows()
cap.release()'''

import cv2
from cvzone.PoseModule import PoseDetector
import numpy as np

detector = PoseDetector() 
#img=cv2.imread('C:/Users/kesav/Pictures/OIP.jpg')

cap=cv2.VideoCapture(1)
while cap.isOpened():
    _,img=cap.read()
    img = detector.findPose(img)
    imlist, bbox = detector.findPosition(img,bboxWithHands=True)
    measure_list=[]
    b=bbox.items()
    
    for i in b:
        if len(i[1])==2:
            x=i[1][0]
            y=i[1][0]
            cv2.circle(img,i[1],10,(0,0,255),-1)
            cv2.putText(img,'width :{}'.format(round(measure_list[2],1)),(10,40),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
            cv2.putText(img,'height :{}'.format(round(measure_list[3],1)),(10,80),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
            #print(i[1])
            
        elif len(i[1])>2:
            for n in i[1]:
                measure_list.append(n)
                #print(n)
            #print(measure_list)
            cv2.rectangle(img,(measure_list[0],measure_list[1],measure_list[2],measure_list[3]),(0,255,0),10)
        
    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
    
cv2.destroyAllWindows()
cap.release()