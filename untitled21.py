# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 22:36:33 2023

@author: kesav
"""
import cv2
import mediapipe as mp
import time

'''
cap=cv2.VideoCapture(1)
mphands=mp.solutions.hands
mpdraw=mp.solutions.drawing_utils
pTime=0
cTime=0

hands=mphands.Hands(static_image_mode=False,max_num_hands=1,min_detection_confidence=0.1)
while cap.isOpened():
    _,frame=cap.read()
    imgRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results=hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            for id,lm in enumerate(handlms.landmark):
                #print(id,lm)
                h,w,c=frame.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                #print(id,cx,cy)
                if id==14:
                    cv2.circle(frame,(cx,cy),25,(255,0,255),cv2.FILLED)
            mpdraw.draw_landmarks(frame,handlms,mphands.HAND_CONNECTIONS)
            
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(frame,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
    
cv2.destroyAllWindows()
cap.release()
'''

mphands=mp.solutions.hands
hands=mphands.Hands()
mpdraw=mp.solutions.drawing_utils

cap=cv2.VideoCapture(1)

while cap.isOpened():
    _,img=cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result=hands.process(imgRGB)
    #print(result.multi_hand_landmarks)
    if result.multi_hand_landmarks:
        for handlms in result.multi_hand_landmarks:
            #print(handlms)
            mpdraw.draw_landmarks(img,handlms,mphands.HAND_CONNECTIONS)
    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
    
cv2.destroyAllWindows()
cap.release()
'''
class handDetector():
    def __init__(self,node=False,maxHands=2,detectionCon=0.5,trackCon=0.5):
        self.node=node
        self.maxHands=maxHands
        self.detectionCon=detectionCon
        self.trackCon=trackCon
        
        self.mphands=mp.solutions.hands
        self.hands=self.mphands.Hands()
        self.mpdraw=mp.solutions.drawing_utils
        
    def findHands(self,img,draw=True):
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        result=self.hands.process(imgRGB)
        if result.multi_hand_landmarks:
            for handlms in result.multi_hand_landmarks:                        
                self.mpdraw.draw_landmarks(img,handlms,self.mphands.HAND_CONNECTIONS)
                
        return img
        
    def findPosition(self,img,handNo=8,draw=True):
        imlist=[]
        if results.multi_hand_landmarks:
            myHand=self.result.landmark.multi_hand_landmarks[handNo]
            for id,lm in enumerate(handlms.landmark):
                                #print(id,lm)
                                h,w,c=img.shape
                                cx,cy=int(lm.x*w),int(lm.y*h)
                                imlist.apend([id,cx,cy])
                                #print(cx,cy)
                                #if id==14:
                                if draw:
                                    cv2.circle(img,(cx,cy),25,(255,0,0),-1)
        return imlist


def main():
    cTime=0
    pTime=0
    cap=cv2.VideoCapture(1)
    detector=handDetector()
    while cap.isOpened():
        _,img=cap.read()
        img=detector.findHands(img)
        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime
        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        cv2.imshow('frame',img)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    cv2.destroyAllWindows()


main()
'''