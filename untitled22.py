# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 22:34:17 2023

@author: kesav
"""

import cv2
import mediapipe as mp
import time
'''
mppose=mp.solutions.pose
pose=mppose.Pose()
mpdraw=mp.solutions.drawing_utils

pTime=0
cTime=0
cap=cv2.VideoCapture(1)

while cap.isOpened():
    _,img=cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result=pose.process(imgRGB)
    if result.pose_landmarks:
        mpdraw.draw_landmarks(img,result.pose_landmarks,mppose.POSE_CONNECTIONS)
        #print(result.pose_landmarks)
        for id,lm in enumerate((result.pose_landmarks.landmark)):
            h,w,c=img.shape
            #print(id,lm)
            cx,cy=int(lm.x*w),int(lm.y*h)
            cv2.putText(img,str(id),(cx,cy),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),1)
            #cv2.circle(img,(cx,cy),10,(255,0,0),-1)
            
        
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime    
    cv2.putText(img,str(int(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
    
cv2.destroyAllWindows()
cap.release()'''
