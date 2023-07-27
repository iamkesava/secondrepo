import cv2
import imutils
import time
'''
img=cv2.imread('C:/Users/kesav/Pictures/fruits.jpg')
#cv2.imwrite('fr.jpg',img)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#reimg=imutils.resize(img,width=1000,height=1000)
#blur=cv2.GaussianBlur(img,(25,25),0)
#thersh=cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
#cv2.imshow('frame',cv2.rectangle(img,(10,40),(50,80),(255,0,0),5))
#cv2.imshow('frame',cv2.putText(img,"open",(50,80),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA))
cv2.findContours()
#cv2.imshow('frame',thersh[1])
#print(thersh[0])
#cv2.imshow('frame',gray)
cv2.waitKey(0)'''


'''
vc=cv2.VideoCapture(1)

while True:
    _,frame=vc.read()
    cv2.rectangle(frame,(10,50),(60,100),(0,0,255),5)
    cv2.putText(frame,'Video',(50,80),cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,0),2,cv2.LINE_AA)
    cv2.imshow('frame',frame)
    key=cv2.waitKey(1) &  0xFF
    if key==ord('q'):
        break
    
vc.release()
cv2.destroyAllWindows()
'''

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
    key=cv2.waitKey(1) & 0xFF
    if key==ord('q'):
        break
    
vr.release()
cv2.destroyAllWindows()
'''

img=cv2.imread('C:/Users/kesav/Pictures/fruits.jpg')
reimg=imutils.resize(img,width=500,height=500)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur=cv2.GaussianBlur(img,(101,101),1)
cv2.rectangle(img,(10,40),(50,90),(0,255,255),0)
cv2.putText(img,'Hello',(50,40),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,255),2)
thresh=cv2.threshold(gray,180,255,cv2.THRESH_BINARY)[1]
contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
cv2.imshow('frame',img)

cv2.waitKey(0)
'''
img = cv2.imread('C:/Users/kesav/Pictures/fruits.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)[1]
contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

# Display the image with contours
cv2.imshow('Contours', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

'''
import cv2
import numpy as np

# Let's load a simple image with 3 black squares
image = cv2.imread('C://Users//gfg//shapes.jpg')
cv2.waitKey(0)

# Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Find Canny edges
edged = cv2.Canny(gray, 30, 200)
cv2.waitKey(0)

# Finding Contours
# Use a copy of the image e.g. edged.copy()
# since findContours alters the image
contours, hierarchy = cv2.findContours(edged,
	cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

cv2.imshow('Canny Edges After Contouring', edged)
cv2.waitKey(0)

print("Number of Contours found = " + str(len(contours)))

# Draw all contours
# -1 signifies drawing all contours
cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

cv2.imshow('Contours', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''

