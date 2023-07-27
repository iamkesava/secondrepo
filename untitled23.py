# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 11:33:33 2023

@author: kesav
"""

###############################################




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
    print(b)
    
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
'''

'''
import cv2
import mediapipe as mp
import time

mppose=mp.solutions.pose
pose=mppose.Pose()
mpdraw=mp.solutions.drawing_utils

pTime=0
cTime=0
cap=cv2.VideoCapture(1)

while cap.isOpened():
    _,img=cap.read()
    img2=img.copy()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result=pose.process(imgRGB)
    if result.pose_landmarks:
        mpdraw.draw_landmarks(img,result.pose_landmarks,mppose.POSE_CONNECTIONS)

        right_hand_ankle_x=0
        right_hand_ankle_y=0
        right_hand_ankle=0
        
        left_hand_ankle_x=0
        left_hand_ankle_y=0
        left_hand_ankle=0
        
        right_shoulder_x=0
        right_shoulder_y=0
        right_shoulder=0
        
        left_shoulder_x=0
        left_shoulder_y=0
        left_shoulder=0
        
        right_leg_anckle_x=0
        right_leg_anckle_y=0
        right_leg_anckle=0
        
        right_leg_knee_x=0
        right_leg_knee_y=0
        right_leg_knee=0
        
        left_leg_anckle_x=0
        left_leg_anckle_y=0
        left_leg_anckle=0
        
        right_shoulder_x=0
        right_shoulder_y=0
        right_shoulder=0
        
        left_shoulder_x=0
        left_shoulder_y=0
        left_shoulder=0
        
        right_hand_knee_x=0
        right_hand_knee_y=0
        right_hand_knee=0
        
        left_hand_knee_x=0
        left_hand_knee_y=0
        left_hand_knee=0
        
        left_leg_knee_x=0
        left_leg_knee_y=0
        left_leg_knee=0
        
        right_hip_x=0
        right_hip_y=0
        right_hip=0

        left_hip_x=0
        left_hip_y=0
        left_hip=0
        
        alt_img=0
    
        for id,lm in enumerate((result.pose_landmarks.landmark)):     
            h,w,c=img.shape
            #print(id,lm)
            cx,cy=int(lm.x*w),int(lm.y*h)
            cv2.putText(img,str(id),(cx,cy),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),1)
            #cv2.circle(img,(cx,cy),10,(255,0,0),-1)
            if id==11:
                left_shoulder_x=cx
                left_shoulder_y=cy
                left_shoulder=(cx,cy)
                
               
            #right shoulder
            elif id==12:
                #y=cy-h
                right_shoulder_x=cx
                right_shoulder_y=cy
                right_shoulder=(cx,cy)
                
            #right hand anckle
            elif id==16:
                right_hand_ankle_x=cx
                right_hand_ankle_y=cy
                right_hand_ankle=(cx,cy)
                
            #left hand anckle
            elif id==15:
                left_hand_ankle_x=cx
                left_hand_ankle_y=cy
                left_hand_ankle=(cx,cy)
                               
            #left hip
            elif id==23:
                left_hip_x=cx
                left_hip_y=cy
                left_hip=(cx,cy)
                
            #right hip
            elif id==24:
                right_hip_x=cx
                right_hip_y=cy
                right_hip=(cx,cy)
                
            #left leg anckle
            elif id==27:
                left_leg_anckle_x=cx
                left_leg_anckle_y=cy
                left_leg_anckle=(cx,cy)
              
            #right leg anckle
            elif id==28:
                right_leg_anckle_x=cx
                right_leg_anckle_y=cy
                right_leg_anckle=(cx,cy)
                
            #right leg knee
            elif id==26:
                right_leg_knee_x=cx
                right_leg_knee_y=cy
                right_leg_knee=(cx,cy)
                
            #left leg knee
            elif id==25:
                left_leg_knee_x=cx
                left_leg_knee_y=cy
                left_leg_knee=(cx,cy)
                
            #right hand knee
            elif id==14:
                right_hand_knee_x=cx
                right_hand_knee_y=cy
                right_hand_knee=(cx,cy)
                
            #left hand knee
            elif id==13:
                left_hand_knee_x=cx
                left_hand_knee_y=cy
                left_hand_knee=(cx,cy)
         
            if left_shoulder and right_hip:
                cv2.line(img2,(right_shoulder),(left_shoulder),(0,255,0),2)
                cv2.line(img2,(left_shoulder),(left_hip),(0,255,0),2)
                cv2.line(img2,(left_hip),(right_hip),(0,255,0),2)
                cv2.line(img2,(right_hip),(right_shoulder),(0,255,0),2)
                
            if right_shoulder and right_hand_knee:
                x = min(right_shoulder[0], right_hand_knee[0])
                y = min(right_shoulder[1], right_hand_knee[1])
                w = abs(right_shoulder[0] - right_hand_knee[0])
                h = abs(right_shoulder[1] - right_hand_knee[1])
                #cv2.putText(img2,str(h),(50,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2,cv2.LINE_AA)
                #cv2.circle(img,(int(x),int(y),5,(0,0,255,-1))
                cv2.rectangle(img2,right_shoulder,right_hand_knee,(255,0,0),2)
                
            if right_hand_knee and right_hand_ankle:
                cv2.rectangle(img2,right_hand_knee,right_hand_ankle,(255,0,0),2)
                x = min(right_shoulder[0], right_hand_knee[0])
                y = min(right_shoulder[1], right_hand_knee[1])
                w = abs(right_shoulder[0] - right_hand_knee[0])
                h = abs(right_shoulder[1] - right_hand_knee[1])
                cv2.putText(img2,str(h),(50,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2,cv2.LINE_AA)
            
            if left_shoulder and left_hand_knee:
                cv2.rectangle(img2,left_shoulder,left_hand_knee,(255,0,0),2)
                
            if left_hand_knee and left_hand_ankle:
                cv2.rectangle(img2,left_hand_knee,left_hand_ankle,(255,0,0),2)
                
            if right_hip and right_leg_knee:
                cv2.rectangle(img2,right_hip,right_leg_knee,(255,0,0),2)
                
            if right_leg_knee and right_leg_anckle:
                cv2.rectangle(img2,right_leg_knee,right_leg_anckle,(255,0,0),2)
                
            if left_hip and left_leg_knee:
                cv2.rectangle(img2,left_hip,left_leg_knee,(255,0,0),2)
                
            if left_leg_knee and left_leg_anckle:
                cv2.rectangle(img2,left_hip,left_leg_anckle,(255,0,0),2)
        
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime    
    cv2.putText(img,str(int(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
    cv2.imshow('img',img2)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
    
cv2.destroyAllWindows()
cap.release()
'''

'''
import cv2
import mediapipe as mp
import time
import numpy as np

mppose=mp.solutions.pose
pose=mppose.Pose()
mpdraw=mp.solutions.drawing_utils

cap=cv2.VideoCapture(1)
while True:
    _,img=cap.read()
    img2=img.copy()
    npimg=np.zeros((700,700,3),np.uint8)
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result=pose.process(imgRGB)
    if result.pose_landmarks:
        mpdraw.draw_landmarks(img,result.pose_landmarks,mppose.POSE_CONNECTIONS)
    
        right_hand_ankle_x=0
        right_hand_ankle_y=0
        right_hand_ankle=0
            
        left_hand_ankle_x=0
        left_hand_ankle_y=0
        left_hand_ankle=0
            
        right_shoulder_x=0
        right_shoulder_y=0
        right_shoulder=0
            
        left_shoulder_x=0
        left_shoulder_y=0
        left_shoulder=0
            
        right_leg_anckle_x=0
        right_leg_anckle_y=0
        right_leg_anckle=0
        
        right_leg_knee_x=0
        right_leg_knee_y=0
        right_leg_knee=0
            
        left_leg_anckle_x=0
        left_leg_anckle_y=0
        left_leg_anckle=0
        
        right_shoulder_x=0
        right_shoulder_y=0
        right_shoulder=0
        
        left_shoulder_x=0
        left_shoulder_y=0
        left_shoulder=0
        
        right_hand_knee_x=0
        right_hand_knee_y=0
        right_hand_knee=0
        
        left_hand_knee_x=0
        left_hand_knee_y=0
        left_hand_knee=0
        
        left_leg_knee_x=0
        left_leg_knee_y=0
        left_leg_knee=0
       
        right_hip_x=0
        right_hip_y=0
        right_hip=0
    
        left_hip_x=0
        left_hip_y=0
        left_hip=0
        
        for id,lm in enumerate((result.pose_landmarks.landmark)):     
           h,w,c=img.shape
           #print(id,lm)
           cx,cy=int(lm.x*w),int(lm.y*h)
           cv2.putText(img,str(id),(cx,cy),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),1)
                #cv2.circle(img,(cx,cy),10,(255,0,0),-1)
           if id==11:
                left_shoulder_x=cx
                left_shoulder_y=cy
                left_shoulder=(cx,cy)              
                   
                #right shoulder
           elif id==12:
                    #y=cy-h
                right_shoulder_x=cx
                right_shoulder_y=cy
                right_shoulder=(cx,cy)
                    
                #right hand anckle
           elif id==16:
                right_hand_ankle_x=cx
                right_hand_ankle_y=cy
                right_hand_ankle=(cx,cy)
                    
                #left hand anckle
           elif id==15:
                left_hand_ankle_x=cx
                left_hand_ankle_y=cy
                left_hand_ankle=(cx,cy)
                                   
                #left hip
           elif id==23:
                left_hip_x=cx
                left_hip_y=cy
                left_hip=(cx,cy)
                    
                #right hip
           elif id==24:
                right_hip_x=cx
                right_hip_y=cy
                right_hip=(cx,cy)
                    
                #left leg anckle
           elif id==27:
                left_leg_anckle_x=cx
                left_leg_anckle_y=cy
                left_leg_anckle=(cx,cy)
                  
                #right leg anckle
           elif id==28:
                right_leg_anckle_x=cx
                right_leg_anckle_y=cy
                right_leg_anckle=(cx,cy)
                
            #right leg knee
           elif id==26:
                right_leg_knee_x=cx
                right_leg_knee_y=cy
                right_leg_knee=(cx,cy)
                
            #left leg knee
           elif id==25:
                left_leg_knee_x=cx
                left_leg_knee_y=cy
                left_leg_knee=(cx,cy)
                
            #right hand knee
           elif id==14:
                right_hand_knee_x=cx
                right_hand_knee_y=cy
                right_hand_knee=(cx,cy)
                
            #left hand knee
           elif id==13:
                left_hand_knee_x=cx
                left_hand_knee_y=cy
                left_hand_knee=(cx,cy)
            
           if left_shoulder and right_hip:
                cv2.line(img2,(right_shoulder),(left_shoulder),(0,255,0),2)
                cv2.line(img2,(left_shoulder),(left_hip),(0,255,0),2)
                cv2.line(img2,(left_hip),(right_hip),(0,255,0),2)
                cv2.line(img2,(right_hip),(right_shoulder),(0,255,0),2)
                
           if right_shoulder and right_hand_knee:
                x = min(right_shoulder[0], right_hand_knee[0])
                y = min(right_shoulder[1], right_hand_knee[1])
                w = abs(right_shoulder[0] - right_hand_knee[0])
                h = abs(right_shoulder[1] - right_hand_knee[1])
                text=f'right shoulder to right hand knee:{h/5}'
                cv2.putText(npimg,text,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv2.LINE_AA)
                #cv2.circle(img,(int(x),int(y),5,(0,0,255,-1))
                cv2.rectangle(img2,right_shoulder,right_hand_knee,(255,0,0),2)
                
           if right_hand_knee and right_hand_ankle:
                cv2.rectangle(img2,right_hand_knee,right_hand_ankle,(255,0,0),2)
                x = min(right_hand_knee[0], right_hand_ankle[0])
                y = min(right_hand_knee[1], right_hand_ankle[1])
                w = abs(right_hand_knee[0] - right_hand_ankle[0])
                h = abs(right_hand_knee[1] - right_hand_ankle[1])
                text=f'right hand knee to right hand ankle:{h/5}'
                cv2.putText(npimg,str(h),(50,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv2.LINE_AA)
               
           if left_shoulder and left_hand_knee:
               x = min(left_shoulder[0], left_hand_knee[0])
               y = min(left_shoulder[1], left_hand_knee[1])
               w = abs(left_shoulder[0] - left_hand_knee[0])
               h = abs(left_shoulder[1] - left_hand_knee[1])
               text=f'left shoulder to left hand_knee:{h/5}'
               cv2.putText(npimg,text,(50,150),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv2.LINE_AA)
               cv2.rectangle(img2,left_shoulder,left_hand_knee,(255,0,0),2)
                
           if left_hand_knee and left_hand_ankle:
               x = min(left_hand_knee[0], left_hand_ankle[0])
               y = min(left_hand_knee[1], left_hand_ankle[1])
               w = abs(left_hand_knee[0] - left_hand_ankle[0])
               h = abs(left_hand_knee[1] - left_hand_ankle[1])
               text=f'left hand knee to left hand ankle:{h/5}'
               cv2.putText(npimg,text,(50,200),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv2.LINE_AA)
               cv2.rectangle(img2,left_hand_knee,left_hand_ankle,(255,0,0),2)
                
           if right_hip and right_leg_knee:
               x = min(right_hip[0], right_leg_knee[0])
               y = min(right_hip[1], right_leg_knee[1])
               w = abs(right_hip[0] - right_leg_knee[0])
               h = abs(right_hip[1] - right_leg_knee[1])
               text=f'right hip to right leg knee:{h/5}'
               cv2.putText(npimg,text,(50,250),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv2.LINE_AA)
               cv2.rectangle(img2,right_hip,right_leg_knee,(255,0,0),2)
                
           if right_leg_knee and right_leg_anckle:
               x = min(right_leg_knee[0], right_leg_knee[0])
               y = min(right_leg_knee[1], right_leg_knee[1])
               w = abs(right_leg_knee[0] - right_leg_knee[0])
               h = abs(right_leg_knee[1] - right_leg_knee[1])
               text=f'right leg knee to right leg anckle:{h/5}'
               cv2.putText(npimg,text,(50,300),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv2.LINE_AA)
               cv2.rectangle(img2,right_leg_knee,right_leg_anckle,(255,0,0),2)
                
           if left_hip and left_leg_knee:
               x = min(left_hip[0], left_leg_knee[0])
               y = min(left_hip[1], left_leg_knee[1])
               w = abs(left_hip[0] - left_leg_knee[0])
               h = abs(left_hip[1] - left_leg_knee[1])
               text=f'left hip to left leg knee:{h/5}'
               cv2.putText(npimg,text,(50,350),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv2.LINE_AA)
               cv2.rectangle(img2,left_hip,left_leg_knee,(255,0,0),2)
                
           if left_leg_knee and left_leg_anckle:
               x = min(left_leg_knee[0], left_leg_anckle[0])
               y = min(left_leg_knee[1], left_leg_anckle[1])
               w = abs(left_leg_knee[0] - left_leg_anckle[0])
               h = abs(left_leg_knee[1] - left_leg_anckle[1])
               text=f'left leg knee to left leg anckle:{h/5}'
               cv2.putText(npimg,text,(50,400),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv2.LINE_AA)
               cv2.rectangle(img2,left_leg_knee,left_leg_anckle,(255,0,0),2)
                
           cv2.imshow('img',img2)
           cv2.imshow('img1',npimg)
        
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break

        
cap.release()
cv2.destroyAllWindows()
'''

'''
import cv2
import mediapipe as mp
import math
import numpy as np

#def distance_finder(image):
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
    
# Initialize MediaPipe Pose model
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
x=[300,245,200,170,145,130,112,103,93,87,80,75,70,67,62,59,57]
y=[20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
coff=np.polyfit(x,y,2)
#y=Ax^2+Bx+C
    
image=cv2.imread('C:/Users/kesav/Pictures/OIP.jpg')
#cap=cv2.VideoCapture(1)
# Run pose estimation on the image or webcam feed
with mp_pose.Pose() as pose:
 # while cap.isOpened():  # Uncomment this line to use webcam instead
        #ret, frame = cap.read()  # Uncomment this line to use webcam instead
        
# Convert image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Flip the image horizontally for a mirrored view
        image_rgb = cv2.flip(image_rgb, 1)
        
            # Set image as input to the pose estimator
        results = pose.process(image_rgb)
        
            # Draw landmarks on the image
        image_rgb.flags.writeable = True
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))
        
                # Extract shoulder coordinates
            left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
            image_height, image_width, _ = image.shape
            x1, y1 = int(left_shoulder.x * image_width), int(left_shoulder.y * image_height)
            x2, y2 = int(right_shoulder.x * image_width), int(right_shoulder.y * image_height)
            distance=int(math.sqrt((x2 - x1)**2 + (y2 - y1)**2))#int(math.sqrt((x2-x1)*2+(x2-x1)*2))
                #print(distance)
    
            A,B,C=coff
            distanceCM=A+distance**2+B*distance*C
                #print(distance,distanceCM)
            cv2.putText(image_rgb,f'{distance}cm',[50,50],cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
                #print(abs(x2-x1),distance)
                # Print shoulder coordinates
                #print("Left Shoulder - x1:", x1, "y1:", y1)
                #print("Right Shoulder - x2:", x2, "y2:", y2)
        
            # Display the image with landmarks
        cv2.imshow('Pose Estimation', image_rgb)
        cv2.waitKey(0)
        
#cap.release()
cv2.destroyAllWindows()
'''

############
import cv2
import mediapipe as mp
import math
import numpy as np

mppose=mp.solutions.pose
pose=mppose.Pose()
mpdraw=mp.solutions.drawing_utils

x=[300,245,200,170,145,130,112,103,93,87,80,75,70,67,62,59,57]
y=[20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
coff=np.polyfit(x,y,2)

cap=cv2.VideoCapture(1)

while True:
    _,img=cap.read()
    img2=img.copy()
    image=img.copy()
    npimg=np.zeros((700,700,3),np.uint8)
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result=pose.process(imgRGB)
    if result.pose_landmarks:
        mpdraw.draw_landmarks(imgRGB, result.pose_landmarks, mppose.POSE_CONNECTIONS,
                                          mpdraw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                                          mpdraw.DrawingSpec(color=(0, 255, 0), thickness=2))
        
                # Extract shoulder coordinates
        left_shoulder = result.pose_landmarks.landmark[mppose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = result.pose_landmarks.landmark[mppose.PoseLandmark.RIGHT_SHOULDER]
        
        image_height, image_width, _ = image.shape
        x1, y1 = int(left_shoulder.x * image_width), int(left_shoulder.y * image_height)
        x2, y2 = int(right_shoulder.x * image_width), int(right_shoulder.y * image_height)
        distance=int(math.sqrt((x2 - x1)**2 + (y2 - y1)**2))#int(math.sqrt((x2-x1)*2+(x2-x1)*2))
                #print(distance)
    
        A,B,C=coff
        distanceCM=A+distance**2+B*distance*C
                #print(distance,distanceCM)
                
        if distance>155:
            cv2.putText(img2,f'{distance}cm',[50,50],cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
            
        elif distance<155:
                cv2.putText(img2,f'{distance}cm',[50,50],cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
                
        elif distance<165:
            cv2.putText(img2,f'{distance}cm',[50,50],cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
            
        right_hand_ankle_x=0
        right_hand_ankle_y=0
        right_hand_ankle=0
            
        left_hand_ankle_x=0
        left_hand_ankle_y=0
        left_hand_ankle=0
            
        right_shoulder_x=0
        right_shoulder_y=0
        right_shoulder=0
            
        left_shoulder_x=0
        left_shoulder_y=0
        left_shoulder=0
            
        right_leg_anckle_x=0
        right_leg_anckle_y=0
        right_leg_anckle=0
        
        right_leg_knee_x=0
        right_leg_knee_y=0
        right_leg_knee=0
            
        left_leg_anckle_x=0
        left_leg_anckle_y=0
        left_leg_anckle=0
        
        right_shoulder_x=0
        right_shoulder_y=0
        right_shoulder=0
        
        left_shoulder_x=0
        left_shoulder_y=0
        left_shoulder=0
        
        right_hand_knee_x=0
        right_hand_knee_y=0
        right_hand_knee=0
        
        left_hand_knee_x=0
        left_hand_knee_y=0
        left_hand_knee=0
        
        left_leg_knee_x=0
        left_leg_knee_y=0
        left_leg_knee=0
       
        right_hip_x=0
        right_hip_y=0
        right_hip=0
    
        left_hip_x=0
        left_hip_y=0
        left_hip=0
        
        for id,lm in enumerate((result.pose_landmarks.landmark)):     
           h,w,c=img.shape
           #print(id,lm)
           cx,cy=int(lm.x*w),int(lm.y*h)
           cv2.putText(img,str(id),(cx,cy),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),1)
                #cv2.circle(img,(cx,cy),10,(255,0,0),-1)
           if id==11:
                left_shoulder_x=cx
                left_shoulder_y=cy
                left_shoulder=(cx,cy)              
                   
                #right shoulder
           elif id==12:
                    #y=cy-h
                right_shoulder_x=cx
                right_shoulder_y=cy
                right_shoulder=(cx,cy)
                    
                #right hand anckle
           elif id==16:
                right_hand_ankle_x=cx
                right_hand_ankle_y=cy
                right_hand_ankle=(cx,cy)
                    
                #left hand anckle
           elif id==15:
                left_hand_ankle_x=cx
                left_hand_ankle_y=cy
                left_hand_ankle=(cx,cy)
                                   
                #left hip
           elif id==23:
                left_hip_x=cx
                left_hip_y=cy
                left_hip=(cx,cy)
                    
                #right hip
           elif id==24:
                right_hip_x=cx
                right_hip_y=cy
                right_hip=(cx,cy)
                    
                #left leg anckle
           elif id==27:
                left_leg_anckle_x=cx
                left_leg_anckle_y=cy
                left_leg_anckle=(cx,cy)
                  
                #right leg anckle
           elif id==28:
                right_leg_anckle_x=cx
                right_leg_anckle_y=cy
                right_leg_anckle=(cx,cy)
                
            #right leg knee
           elif id==26:
                right_leg_knee_x=cx
                right_leg_knee_y=cy
                right_leg_knee=(cx,cy)
                
            #left leg knee
           elif id==25:
                left_leg_knee_x=cx
                left_leg_knee_y=cy
                left_leg_knee=(cx,cy)
                
            #right hand knee
           elif id==14:
                right_hand_knee_x=cx
                right_hand_knee_y=cy
                right_hand_knee=(cx,cy)
                
            #left hand knee
           elif id==13:
                left_hand_knee_x=cx
                left_hand_knee_y=cy
                left_hand_knee=(cx,cy)
            
           if left_shoulder and right_hip:
                cv2.line(img2,(right_shoulder),(left_shoulder),(0,255,0),2)
                cv2.line(img2,(left_shoulder),(left_hip),(0,255,0),2)
                cv2.line(img2,(left_hip),(right_hip),(0,255,0),2)
                cv2.line(img2,(right_hip),(right_shoulder),(0,255,0),2)
                
           if right_shoulder and right_hand_knee:
                x = min(right_shoulder[0], right_hand_knee[0])
                y = min(right_shoulder[1], right_hand_knee[1])
                w = abs(right_shoulder[0] - right_hand_knee[0])
                h = abs(right_shoulder[1] - right_hand_knee[1])
                text=f'right shoulder to right hand knee:{h/5}'
                cv2.putText(npimg,text,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv2.LINE_AA)
                #cv2.circle(img,(int(x),int(y),5,(0,0,255,-1))
                cv2.rectangle(img2,right_shoulder,right_hand_knee,(255,0,0),2)
                
           if right_hand_knee and right_hand_ankle:
                cv2.rectangle(img2,right_hand_knee,right_hand_ankle,(255,0,0),2)
                x = min(right_hand_knee[0], right_hand_ankle[0])
                y = min(right_hand_knee[1], right_hand_ankle[1])
                w = abs(right_hand_knee[0] - right_hand_ankle[0])
                h = abs(right_hand_knee[1] - right_hand_ankle[1])
                text=f'right hand knee to right hand ankle:{h/5}'
                cv2.putText(npimg,str(h),(50,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv2.LINE_AA)
               
           if left_shoulder and left_hand_knee:
               x = min(left_shoulder[0], left_hand_knee[0])
               y = min(left_shoulder[1], left_hand_knee[1])
               w = abs(left_shoulder[0] - left_hand_knee[0])
               h = abs(left_shoulder[1] - left_hand_knee[1])
               text=f'left shoulder to left hand_knee:{h/5}'
               cv2.putText(npimg,text,(50,150),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv2.LINE_AA)
               cv2.rectangle(img2,left_shoulder,left_hand_knee,(255,0,0),2)
                
           if left_hand_knee and left_hand_ankle:
               x = min(left_hand_knee[0], left_hand_ankle[0])
               y = min(left_hand_knee[1], left_hand_ankle[1])
               w = abs(left_hand_knee[0] - left_hand_ankle[0])
               h = abs(left_hand_knee[1] - left_hand_ankle[1])
               text=f'left hand knee to left hand ankle:{h/5}'
               cv2.putText(npimg,text,(50,200),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv2.LINE_AA)
               cv2.rectangle(img2,left_hand_knee,left_hand_ankle,(255,0,0),2)
                
           if right_hip and right_leg_knee:
               x = min(right_hip[0], right_leg_knee[0])
               y = min(right_hip[1], right_leg_knee[1])
               w = abs(right_hip[0] - right_leg_knee[0])
               h = abs(right_hip[1] - right_leg_knee[1])
               text=f'right hip to right leg knee:{h/5}'
               cv2.putText(npimg,text,(50,250),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv2.LINE_AA)
               cv2.rectangle(img2,right_hip,right_leg_knee,(255,0,0),2)
                
           if right_leg_knee and right_leg_anckle:
               x = min(right_leg_knee[0], right_leg_knee[0])
               y = min(right_leg_knee[1], right_leg_knee[1])
               w = abs(right_leg_knee[0] - right_leg_knee[0])
               h = abs(right_leg_knee[1] - right_leg_knee[1])
               text=f'right leg knee to right leg anckle:{h/5}'
               cv2.putText(npimg,text,(50,300),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv2.LINE_AA)
               cv2.rectangle(img2,right_leg_knee,right_leg_anckle,(255,0,0),2)
                
           if left_hip and left_leg_knee:
               x = min(left_hip[0], left_leg_knee[0])
               y = min(left_hip[1], left_leg_knee[1])
               w = abs(left_hip[0] - left_leg_knee[0])
               h = abs(left_hip[1] - left_leg_knee[1])
               text=f'left hip to left leg knee:{h/5}'
               cv2.putText(npimg,text,(50,350),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv2.LINE_AA)
               cv2.rectangle(img2,left_hip,left_leg_knee,(255,0,0),2)
                
           if left_leg_knee and left_leg_anckle:
               x = min(left_leg_knee[0], left_leg_anckle[0])
               y = min(left_leg_knee[1], left_leg_anckle[1])
               w = abs(left_leg_knee[0] - left_leg_anckle[0])
               h = abs(left_leg_knee[1] - left_leg_anckle[1])
               text=f'left leg knee to left leg anckle:{h/5}'
               cv2.putText(npimg,text,(50,400),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv2.LINE_AA)
               cv2.rectangle(img2,left_leg_knee,left_leg_anckle,(255,0,0),2)
                
           cv2.imshow('img',img2)
           cv2.imshow('img1',npimg)
        
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break

        
cap.release()
cv2.destroyAllWindows()
