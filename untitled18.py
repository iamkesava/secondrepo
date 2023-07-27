# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 22:18:35 2023

@author: kesav
"""
import cv2
import numpy as np
'''
img=cv2.imread('C:/Users/kesav/Pictures/fruits.jpg')
img_big=cv2.imread('C:/Users/kesav/Pictures/numberplate.webp')
print(img.shape)
print(img.size)
print(img.dtype)
b,g,r=cv2.split((img))
img=cv2.merge((b,g,r))
#ball=img[280:340,330:390]
#img[273:333,100:160]=ball
img1=cv2.resize(img,(500,500))
img2=cv2.resize(img_big,(500,500))
img3=cv2.addWeighted(img1,.5,img2,.5,0)
cv2.imshow('image',img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
##########################################
'''
def nothing(x):
    print(x)
    
    
img=np.zeros((300,512,3),np.uint8)
cv2.namedWindow('image')
cv2.createTrackbar('B','image',0,255,nothing)
switch='color/gray'
cv2.createTrackbar(switch,'image',0,1,nothing)

while (1):
    cv2.imshow('image',img)
    
    k=cv2.waitKey(1) & 0xFF 
    if k==27:
        break
    
    pos=cv2.getTrackbarPos('B','image')
    font=cv2.FONT_HERSHEY_SIMPLEX
    font=cv2.putText(img,str(pos),(50,150),font,4,(0,0,255))
    
    if s==0:
        img[:]=0
    else:
        img[:]=[b,g,r]

#cv2.waitKey(0)
destroyAllWindows()
'''
'''
import cv2
import imutils

hog=cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
cap=cv2.VideoCapture(1)
while True:
    rect,image=cap.read()
    if rect:
        image=imutils.resize(image,width=min(400,image.shape[1]))
        
        (regions,_)=hog.detectMultiScale(image,winStride=(4,4),padding=(4,4),scalar=1.05)
        for (x,y,w,h) in regions:
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.imshow('image',image)
    key=cv2.waitKey(1) & 0xFF
    if key==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()'''
'''
cap=cv2.VideoCapture(1)

while cap.isOpened():
    _,frame=cap.read()
    #print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow('image',grey)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
    
cv2.destroyAllWindows()
cap.release()
'''

'''
#img = cv2.imread('C:/Users/kesav/Pictures/fruits.jpg')
img=np.zeros((512,512,3),dtype=np.uint8)
# Draw a line on the image
img = cv2.line(img, (0, 0), (255, 255), (0, 255, 255), 2)
img=cv2.arrowedLine(img,(255,255),(100,0),(0,255,0),2)
img=cv2.rectangle(img,(10,100),(200,10),(255,0,0),-1)
img=cv2.circle(img,(107,63),63,(0,0,255),-1)

# Display the image
cv2.imshow('image', img)
cv2.waitKey(0)
'''
'''
cap=cv2.VideoCapture(1)

cap.set(3,1208)
cap.set(4,720)
while cap.isOpened():
    _,frame=cap.read()
    grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    text='height:''{cap.get(3)}\nwidth:{cap.get(4)}'
    grey=cv2.putText(grey,text(50,100),cv2.FONT_HIRERACHY_SIMPLEX,(0,255,0),2,cv2.LINE_AA)
    cv2.imshow('image',grey)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
    
cv2.destroyAllWindows()
cap.release()'''

'''
events=[i for i in dir(cv2) if 'EVENT' in i]
print(events)
'''

'''
def click_event(event,x,y,flags,parm):
    #print(event)
    if event==cv2.EVENT_LBUTTONDOWN:
        #print(f'Events:{event}\nflags:{flags}\nparm:{parm}')
        #print()
        #print(x,',',y)
        font=cv2.FONT_HERSHEY_SIMPLEX
        strXY=str(x)+','+str(y)
        cv2.putText(img,strXY,(x,y),font,1,(255,255,0),2)
        cv2.imshow('image',img)
        
    elif event==cv2.EVENT_RBUTTONDOWN:   
        blue=img[y,x,0]
        green=img[y,x,1]
        red=img[y,x,2]
        img2[:]=[blue,green,red]
        cv2.imshow('image2',img2)
'''
'''
list=[]

def click_event(event,x,y,flags,parm):
    if event==cv2.EVENT_LBUTTONDOWN:          
        cv2.circle(img,(x,y),5,(0,255,0),-1) 
        list.append((x,y))
        
        if len(list)>=2:
            cv2.line(img,list[-1],list[-2],(0,255,0),2)
            #list.clear()
        cv2.imshow('image',img)       

               
    
img=np.zeros((512,512,3),np.uint8)

cv2.imshow('image',img)

cv2.setMouseCallback('image',click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()'''
'''
img=cv2.imread('C:/Users/kesav/Pictures/fruits.jpg')
b,g,r=cv2.split(img)
cv2.imshow('image1',b)
cv2.imshow('image2',g)
cv2.imshow('image3',r)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
'''
a=cv2.resize(cv2.imread('C:/Users/kesav/Pictures/fruits.jpg'),(512,512))
b=cv2.resize(cv2.imread('C:/Users/kesav/Pictures/OIP.jpg'),(512,512))
cv2.imshow('image',cv2.add(a,b))
cv2.imshow('image',cv2.addWeighted(a,.3,b,.7,0))
cv2.waitKey(0)
cv2.destroyAllWindows()'''

'''
def nothing(x):   
    print(x)

img=np.zeros((512,512,3),np.uint8)
cv2.namedWindow('image')

cv2.createTrackbar('B','image',0,255,nothing)
cv2.createTrackbar('G','image',0,255,nothing)
cv2.createTrackbar('R','image',0,255,nothing)

switch='0:OFF\n1:ON'
cv2.createTrackbar(switch,'image',0,1,nothing)

while True:
    cv2.imshow('image',img)
    k=cv2.waitKey(1) & 0xFF
    if k==27:
        break
    b=cv2.getTrackbarPos('B','image')
    g=cv2.getTrackbarPos('G','image')
    r=cv2.getTrackbarPos('R','image')
    s=cv2.getTrackbarPos(switch,'image')
    
    if s==0:
        img[:]=0
        
    else:
        img[:]=[b,g,r]

cv2.destroyAllWindows()
'''
'''
img1=np.zeros((250,500,3),np.uint8)
img1=cv2.rectangle(img1,(200,0),(300,100),(255,255,255),-1)
img2=cv2.imread('C:/Users/kesav/Pictures/download.jpg')
img2=cv2.resize(img2,(500,250))

# A and B should be 1
#bitAnd=cv2.bitwise_and(img2,img1)

#bitOr=cv2.bitwise_or(img2,img1)

#bitxor=cv2.bitwise_xor(img2,img1)

bitNot1=cv2.bitwise_not(img1)

bitNot2=cv2.bitwise_not(img2)

cv2.imshow('image1',img1)
cv2.imshow('image2',img2)
cv2.imshow('image3',bitNot1)
cv2.imshow('image4',bitNot2)

cv2.waitKey(0)
'''

'''
def nothing(x):
    pass

cap=cv2.VideoCapture(1)

cv2.namedWindow('Trackbar')
cv2.createTrackbar('LH','Trackbar',0,255,nothing)
cv2.createTrackbar('LS','Trackbar',0,255,nothing)
cv2.createTrackbar('LV','Trackbar',0,255,nothing)
cv2.createTrackbar('UH','Trackbar',255,255,nothing)
cv2.createTrackbar('US','Trackbar',255,255,nothing)
cv2.createTrackbar('UV','Trackbar',225,255,nothing)

cap.set(3,250)
cap.set(4,250)

while True:
    _,frame=cap.read()
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
    l_h=cv2.getTrackbarPos('LH','Trackbar')
    l_s=cv2.getTrackbarPos('LS','Trackbar')
    l_v=cv2.getTrackbarPos('LV','Trackbar')
    
    u_h=cv2.getTrackbarPos('UH','Trackbar')
    u_s=cv2.getTrackbarPos('US','Trackbar')
    u_v=cv2.getTrackbarPos('LH','Trackbar')
    
    l_b=np.array([l_h,l_s,l_v])
    u_b=np.array([u_h,u_s,u_v])
    
    mask=cv2.inRange(hsv,l_b,u_b)
    res=cv2.bitwise_and(frame,frame,mask=mask)
    cv2.imshow('mask',mask)
    cv2.imshow('frame',frame)
    cv2.imshow('res',res)
    key=cv2.waitKey(1)
    if key==ord('q'):
        break
    
cv2.destroyAllWindows()
'''
'''
img=cv2.imread('C:/Users/kesav/Pictures/gradient.png')
_,thi=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
_,th2=cv2.threshold(img,200,255,cv2.THRESH_BINARY_INV)
_,th3=cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
_,th4=cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
_,th5=cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)

cv2.imshow('img',img)
cv2.imshow('img2',thi)
cv2.imshow('img3',th2)
cv2.imshow('img5',th3)
cv2.imshow('img5',th4)
cv2.imshow('img6',th5)

cv2.waitKey(0)'''

'''
img=cv2.imread('C:/Users/kesav/Pictures/sodakui2.webp')
_,thi=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
_,thi1=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
cv2.imshow('img',thi)
cv2.imshow('img',thi1)
cv2.waitKey(0)
'''

img1=np.zeros((250,500,3),np.uint8)
img1=cv2.rectangle(img1,(200,0),(300,100),(255,255,255),-1)
img2=cv2.imread('C:/Users/kesav/Pictures/download.jpg')
img2=cv2.resize(img2,(500,250))

cv2.imshow('frame1',img1)
cv2.imshow('frame2',img2)

cv2.waitKey(0)