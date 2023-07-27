# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 19:26:30 2023

@author: kesav
"""
'''
import cv2
from cvzone.PoseModule import PoseDetector

detector=PoseDetector
cap=cv2.VideoCapture(0)
while True:
    success,img=cap.read()
    img=detector.findPose(img)
    imlist,bbox=detector.findPosition(img)
    cv2.imshow('frame',img)
    key=cv2.waitKey(1) & 0xFF
    if key==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
'''

'''
import cv2
from cvzone.PoseModule import PoseDetector

detector = PoseDetector()  # Create an instance of the PoseDetector class
cap = cv2.VideoCapture(1)
while True:
    success, img = cap.read()
    img = detector.findPose(img)
    imlist, bbox = detector.findPosition(img)
    cv2.imshow('frame', img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''
'''
import cv2
from cvzone.PoseModule import PoseDetector

detector = PoseDetector()  # Create an instance of the PoseDetector class
cap = cv2.VideoCapture(1)
while True:
    success, img = cap.read()
    img = detector.findPose(img)
    imlist, bbox = detector.findPosition(img,bboxWithHands=True)
    #print(bbox)
    measure_list=[]
    b=bbox.items()
    for i in b:
        if len(i[1])>2:
            #measure_list.append(i[1])
            for n in i[1]:
                measure_list.append(n)
            x = measure_list[0]
            y = measure_list[1]
            width = measure_list[2] - measure_list[0]
            height = measure_list[3] - measure_list[1]
    
            print("x:", x)
            print("y:", y)
            print("width:", width)
            print("height:", height)
            print('_________________')
            cv2.circle(img,(height,width),10,(0,0,255),-1)

    #print(measure_list) 
    #cv2.rectangle(img, (measure_list[0], measure_list[1]), (measure_list[2], measure_list[3]), (255, 0, 0), 10)
    cv2.imshow('frame', img)
    #measure_list.clear()
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

'''
'''
import cv2
from cvzone.PoseModule import PoseDetector

detector = PoseDetector()  # Create an instance of the PoseDetector class
cap = cv2.VideoCapture(1)
while True:
    success, img = cap.read()
    img = detector.findPose(img)
    imlist, bbox = detector.findPosition(img,bboxWithHands=True)
    measure_list=[]
    b=bbox.items()
    for i in b:
        if len(i[1])>2:
            for n in i[1]:
                measure_list.append(n)
            x = measure_list[0]
            y = measure_list[1]
            width = measure_list[2] - measure_list[0]
            height = measure_list[3] - measure_list[1]

            bbox = [(x, width), (y, height)]   
            x1, y1 = bbox[0]
            x2, y2 = bbox[1]
            
            width = x2 - x1
            height = y2 - y1
            
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            #print("Center X:", center_x)
            #print("Center Y:", center_y)

            cv2.circle(img,(width,height),10,(0,0,255),-1)

    cv2.imshow('frame', img)
    #measure_list.clear()
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()'''

import cv2
from cvzone.PoseModule import PoseDetector

detector = PoseDetector()  # Create an instance of the PoseDetector class
cap = cv2.VideoCapture(1)
while True:
    success, img = cap.read()
    img = detector.findPose(img)
    imlist, bbox = detector.findPosition(img,bboxWithHands=True)
    measure_list=[]
    #edge_list=[]
    b=bbox.items()
    for i in b:
        #print(i)
        if len(i[1])==2:
            #measure_list.append(i)
            print(i[1])
            print('################')
            #cv2.circle(img,i[1],40,(0,0,255),-1)

        elif len(i[1])>2:
            #measure_list.append(i[1])
            for n in i[1]:
                measure_list.append(n)
                print(measure_list)
            x = measure_list[0]
            y = measure_list[1]
            width = measure_list[2] - measure_list[0]
            height = measure_list[3] - measure_list[1]
    
            
    cv2.imshow('frame', img)
    #img2=img[100:400,100:400]
    #cv2.imshow('img',img2)
    #measure_list.clear()
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#print(len([('bbox', (-42, -98, 933, 1844)), ('center', (424, 824))]))