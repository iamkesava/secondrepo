# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 20:20:36 2023

@author: kesav
"""

import cv2

# Load the video file
cap = cv2.VideoCapture('video.mp4')

# Load the number plate detection classifier
plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

# Initialize an empty list to store detected number plates
plates = []

# Define a function to detect number plates in a frame
def detect_plates(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return plates

# Loop through each frame of the video
while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect number plates in the frame
    plates_in_frame = detect_plates(frame)
    
    # Add detected number plates to the list
    if len(plates_in_frame) > 0:
        for plate in plates_in_frame:
            plates.append(plate)
    
    # Display the frame with detected number plates
    for (x, y, w, h) in plates_in_frame:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video file and close all windows
cap.release()
cv2.destroyAllWindows()

# Print the list of detected number plates
print(plates)