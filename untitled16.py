# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 19:44:34 2023

@author: kesav
"""

import cv2
import numpy as np
'''
cap=cv2.VideoCapture(1)
force=cv2.VideoWriter_fourcc(*'XVID')
out=cv2.VideoWriter('output.avi',force,20.0,(640,480))
print(cap.isOpened())
while cap.isOpened():
    _,frame=cap.read()
    if _==True:
        print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out.write(frame)
        #print(cap.get(cv2.CAP_PROF_FRAME_WIDTH))
        cv2.imshow('frame',frame)
        key=cv2.waitKey(1) & 0xFF
        if key==ord('q'):
            break
    else:
        break
        
cv2.destroyAllWindows()
out.release()
cap.release()'''
'''
img=cv2.imread('C:/Users/kesav/Pictures/fruits.jpg')
img=cv2.line(img,(0,0),(250,250),(255,0,0),10)
img=cv2.arrowedLine(img,(0,0),(100,200),(255,250,0),10)
img=cv2.rectangle(img,(90,0),(250,120),(0,0,255),-1)
img=cv2.circle(img,())
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()'''

'''
cap=cv2.VideoCapture(1)
force=cv2.VideoWriter_fourcc(*'XVID')
print(cap.isOpened())
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.set(3,3000)
cap.set(4,3000)
while cap.isOpened():
    _,frame=cap.read()
    if _==True:
        font=cv2.FONT_HERSHEY_COMPLEX
        text='width'+str('width:'cap.get(3))+' Height:'+str(cap.get(4))
        frame=cv2.putText(frame,text,(10,50),font,1,(0,255,255),5)
        cv2.imshow('frame',frame)
        key=cv2.waitKey(1) & 0xFF
        if key==ord('q'):
            break
    else:
        break
    
cv2.destroyAllWindows()

cap.release()'''

#events=[i for i in dir(cv2) if 'EVENT' in i]
#print(events)

'''
eve=['EVENT_FLAG_ALTKEY', 'EVENT_FLAG_CTRLKEY', 'EVENT_FLAG_LBUTTON', 'EVENT_FLAG_MBUTTON', 
     'EVENT_FLAG_RBUTTON', 'EVENT_FLAG_SHIFTKEY', 'EVENT_LBUTTONDBLCLK', 'EVENT_LBUTTONDOWN', 
     'EVENT_LBUTTONUP', 'EVENT_MBUTTONDBLCLK', 'EVENT_MBUTTONDOWN', 'EVENT_MBUTTONUP', 'EVENT_MOUSEHWHEEL', 
     'EVENT_MOUSEMOVE', 'EVENT_MOUSEWHEEL', 'EVENT_RBUTTONDBLCLK', 'EVENT_RBUTTONDOWN', 'EVENT_RBUTTONUP']

def click_event(event,x,y,flags,param):
    if event==cv2.EVENT_LBUTTONDOWN:
        print(x,',',y)
        font=cv2.FONT_HERSHEY_SIMPLEX
        strXY=str(x)+","+str(y)
        cv2.putText(img,strXY,(x,y),font,1,(255,0,0),2)
        cv2.imshow(image,img)
        
img=np.zeros((512,512,3),np.uint8)
cv2.imshow('image',img)
cv2.setMouseCallback('image',click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()'''
#############################################
'''
def click_event(event,x,y,flags,param):  
    if event==cv2.EVENT_LBUTTONDOWN:
        print(x,',',y)
        
        font=cv2.FONT_HERSHEY_SIMPLEX
        strXY=str(x)+","+str(y)
        cv2.putText(img,strXY,(x,y),font,1,(255,0,0),2)
        cv2.imshow('image',img)
        
    if event==cv2.EVENT_RBUTTONDOWN:
        blue=img[y,x,0]
        green=img[x,y,1]
        red=img[y,x,2]
        font=cv2.FONT_HERSHEY_SIMPLEX
        strBGR=str(blue)+','+str(green)+','+str(red)
        cv2.putText(img,strBGR,(x,y),font,.5,(0,255,255),2)
        print(strBGR)
        print(event,x,y,flags,param)
        cv2.imshow('image',img)

        
img=cv2.imread('C:/Users/kesav/Pictures/fruits.jpg')
cv2.imshow('image',img)
cv2.setMouseCallback('image',click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
#############################################
'''
def click_event(event,x,y,flags,param):
    if event==cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img,(x,y),3,(255,0,0),-1)
        points.append((x,y))
        if len(points)>=2:
            cv2.line(img,points[-1],points[-2],(0,255,0),1)
        cv2.imshow('image',img)

        
img=np.zeros((500,500,3),np.uint8)
cv2.imshow('image',img)
points=[]
cv2.setMouseCallback('image',click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()'''
#####################################################
'''
def click_event(event,x,y,flags,param):
    if event==cv2.EVENT_LBUTTONDOWN:
       blue=img[x,y,0]
       green=img[x,y,1]
       red=img[x,y,2]
       scr=np.zeros((500,500,3),np.uint8)
       scr[:]=[blue,green,red]
       cv2.imshow('color',scr)
       print(blue)
       print(green)
       print(red)
       print([blue,green,red])

        
img=cv2.imread('C:/Users/kesav/Pictures/fruits.jpg')
cv2.imshow('image',img)
points=[]
cv2.setMouseCallback('image',click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()'''
##################################################
'''
img=cv2.imread('C:/Users/kesav/Pictures/numberplate1.webp')]
print(img.shape)
print(img.size)
print(img.dtype)
b,g,r=cv2.split((img))
img=cv2.merge((b,g,r))

ball=img[280:340,330:390]
img[273:333,100:160]=ball
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()'''
############################################
'''
img=cv2.imread('C:/Users/kesav/Pictures/209333.jpg')
img_big=cv2.imread('C:/Users/kesav/Pictures/numberplate.webp')
print(img.shape)
print(img.size)
print(img.dtype)
b,g,r=cv2.split((img))
img=cv2.merge((b,g,r))
ball=img[280:340,330:390]
img[273:333,100:160]=ball
img1=cv2.resize(img,(500,500))
img2=cv2.resize(img_big,(500,500))
img3=cv2.add(img1,img2)
cv2.imshow('image',img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
