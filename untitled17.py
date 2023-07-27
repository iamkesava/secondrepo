
'''
import cv2

def measure_object(image, known_width, focal_length):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform object detection (you can use any object detection algorithm here)
    # ...

    # Compute the object's height and width
    # ...

    # Find contours of the object
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding rectangle of the contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Calculate the real dimensions using the focal length
    real_width = (known_width * focal_length) / w
    real_height = (known_width * focal_length) / h

    return real_width, real_height

def main():
    # Set up the camera capture
    cap = cv2.VideoCapture('C:/Users/kesav/Pictures/Doctor_2021_HD.mp4')

    # Set the known width of the object and focal length
    known_width = 10  # in centimeters
    focal_length = 1000  # in pixels

    while True:
        # Read the current frame from the camera
        ret, frame = cap.read()

        # Perform object detection and measurement
        width, height = measure_object(frame, known_width, focal_length)

        # Display the measurements on the frame
        cv2.putText(frame, f"Width: {round(width, 2)} cm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"Height: {round(height, 2)} cm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
      
        # Display the frame
        cv2.imshow("Object Measurement", frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and destroy windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()'''
''' 
import cv2

# Load the pre-trained HOG (Histogram of Oriented Gradients) model for human detection
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Set the minimum height and width of detected bounding boxes
min_width = 80
min_height = 180

# Start the video capture
video_capture = cv2.VideoCapture(1)

while True:
    # Read the video frame
    ret, frame = video_capture.read()

    # Resize the frame to improve performance
    frame = cv2.resize(frame, (640, 480))

    # Detect humans in the frame
    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(4, 4), scale=1.05)

    # Draw bounding boxes around the detected humans and estimate their height
    for (x, y, w, h) in boxes:
        if w >= min_width and h >= min_height:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Estimate the height of the person based on the bounding box size
            height = 2.5 * h  # Assuming an average human height of 2.5 meters

            # Display the height above the bounding box
            cv2.putText(frame, f"Height: {height:.2f} meters", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Human Detection', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the windows
video_capture.release()
cv2.destroyAllWindows()
'''
#################################
'''
import cv2
import imutils

# Load the HOG descriptor
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Open the video stream
video_stream = cv2.VideoCapture(1)

while True:
    # Read the next frame from the video stream
    ret, frame = video_stream.read()

    # Resize the frame for faster processing
    frame = imutils.resize(frame, width=500)

    # Detect humans in the frame using HOG
    (bounding_box_cordinates, weights) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.03)

    # Draw the bounding boxes around the detected humans and estimate their height
    for (x, y, w, h) in bounding_box_cordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        height = round(h * 0.0328084, 2)  # Convert height from pixels to feet
        cv2.putText(frame, f"Height: {height} ft", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with the detected humans and their estimated height
    cv2.imshow("Real-time Human Detection & Counting", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close all windows
video_stream.release()
cv2.destroyAllWindows()'''
#####################################
'''
import math
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import time

# Time
time_ = 0

# Initializing mediapipe pose class.
# mp_pose = mp.solutions.pose
mp_pose = mp.solutions.holistic

mp_drawing_styles = mp.solutions.drawing_styles
# mp_holistic = mp.solutions.holistic

# Setting up the Pose function.
# pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

# Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils 

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([pose, face, left_hand, right_hand])

def extract_keypoints_Pose(results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        return np.concatenate([pose])
    
def extract_keypoints_Pose_(results):
        pose = []
        for i in range(len(results.pose_landmarks.landmark)):
            sample = [results.pose_landmarks.landmark[i].x, results.pose_landmarks.landmark[i].y]
            pose.append(sample)
        pose = np.array(pose)
        pose = pose.reshape((1,*pose.shape))
        return pose

def detectPose(image, pose, display=True):
    '''
    This function performs pose detection on an image.
    Args:
        image: The input image with a prominent person whose pose landmarks needs to be detected.
        pose: The pose setup function required to perform the pose detection.
        display: A boolean value that is if set to true the function displays the original input image, the resultant image, 
                 and the pose landmarks in 3D plot and returns nothing.
    Returns:
        output_image: The input image with the detected pose landmarks drawn.
        landmarks: A list of detected landmarks converted into their original scale.
    '''
    
    # Create a copy of the input image.
    output_image = image.copy()
    
    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform the Pose Detection.
    results = pose.process(imageRGB)
    
    # Retrieve the height and width of the input image.
    height, width, _ = image.shape
    
    # Initialize a list to store the detected landmarks.
    landmarks = []
    
    # keypoint = extract_keypoints_Pose_(results)
    # print(keypoint.shape)
    
    print(len(results.pose_landmarks.landmark))
    
    # Check if any landmarks are detected.
    if results.pose_landmarks:
    
        # Draw Pose landmarks on the output image.
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,connections=mp_pose.POSE_CONNECTIONS,landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        mp_drawing.draw_landmarks(image=output_image,landmark_list=results.left_hand_landmarks,connections=mp_pose.HAND_CONNECTIONS,landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.right_hand_landmarks, connections=mp_pose.HAND_CONNECTIONS,landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
        
        # mp_drawing.draw_landmarks(image=output_image, landmark_list=results.face_landmarks,connections=mp_pose.FACEMESH_CONTOURS,landmark_drawing_spec=None,connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.face_landmarks,connections=mp_pose.FACEMESH_TESSELATION,landmark_drawing_spec=None,connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
        
        # Iterate over the detected landmarks.
        for landmark in results.pose_landmarks.landmark:
            # Append the landmark into the list.
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))
            
        # Append the landmark LEFT Hand into the list. 
        if results.right_hand_landmarks:
            for landmark in results.right_hand_landmarks.landmark:
                # Append the landmark into the list.
                landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                    (landmark.z * width)))
        else:
             for i in range(21):
                # Append the landmark into the list.
                landmarks.append((int(0* width), int(0 * height),
                                    (0 * width)))
        
        # Append the landmark RIGHT Hand into the list. 
        if results.left_hand_landmarks:
            for landmark in results.left_hand_landmarks.landmark:
                # Append the landmark into the list.
                landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                    (landmark.z * width)))
            # print(len(results.left_hand_landmarks.landmark))
        else:
             for i in range(21):
                # Append the landmark into the list.
                landmarks.append((int(0* width), int(0 * height),
                                    (0 * width)))
        
        # print(len(landmarks))
    
    # Check if the original input image and the resultant image are specified to be displayed.
    if display:
    
        # Display the original input image and the resultant image.
        plt.figure(figsize=[22,22])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
        # Also Plot the Pose landmarks in 3D.
        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
        
    # Otherwise
    else:
        
        # Return the output image and the found landmarks.
        return output_image, landmarks
    
def calculateAngle(landmark1, landmark2, landmark3):
    '''
    This function calculates angle between three different landmarks.
    Args:
        landmark1: The first landmark containing the x,y and z coordinates.
        landmark2: The second landmark containing the x,y and z coordinates.
        landmark3: The third landmark containing the x,y and z coordinates.
    Returns:
        angle: The calculated angle between the three landmarks.
    '''

    # Get the required landmarks coordinates.
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    
    # Check if the angle is less than zero.
    if angle < 0:

        # Add 360 to the found angle.
        angle += 360
    
    # Return the calculated angle.
    return angle

def calculateDistance(landmark1, landmark2):
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2

    # Calculate the Distance between the two points
    dis = math.sqrt( ((x2 - x1)**2)+((y2 - y1))**2)

    
    # Return the calculated Distance.
    return dis

left_elbow_angle = 0;
right_elbow_angle = 0;
left_shoulder_angle = 0;
right_shoulder_angle = 0;
left_wrist_angle = 0;
right_wrist_angle = 0;

left_elbow_angle_previous = 0;
right_elbow_angle_previous = 0;
left_shoulder_angle_previous = 0;
right_shoulder_angle_previous = 0;
left_wrist_angle_previous = 0;
right_wrist_angle_previous = 0;

left_elbow_angle_diff = 0;
right_elbow_angle_diff = 0;
left_shoulder_angle_diff = 0;
right_shoulder_angle_diff = 0;
left_wrist_angle_diff = 0;
right_wrist_angle_diff = 0;

Angle_previous = []

def updateAngle_previous():
    # Angle_previous = []
    left_elbow_angle_previous = left_elbow_angle;
    right_elbow_angle_previous = right_elbow_angle;
    left_shoulder_angle_previous = left_shoulder_angle;
    right_shoulder_angle_previous = right_shoulder_angle;
    left_wrist_angle_previous = left_wrist_angle;
    right_wrist_angle_previous = right_wrist_angle;
    Angle_previous.append([left_elbow_angle_previous,right_elbow_angle_previous,left_shoulder_angle_previous,right_shoulder_angle_previous,left_wrist_angle_previous,right_wrist_angle_previous])


Angle_diff = []

def updateAngle_diff():
    # Angle_diff = []
    left_elbow_angle_diff = left_elbow_angle - left_elbow_angle_previous;
    right_elbow_angle_diff = right_elbow_angle - right_elbow_angle_previous;
    left_shoulder_angle_diff = left_shoulder_angle - left_shoulder_angle_previous;
    right_shoulder_angle_diff = right_shoulder_angle - right_shoulder_angle_previous;
    left_wrist_angle_diff = left_wrist_angle - left_wrist_angle_previous;
    right_wrist_angle_diff = right_wrist_angle - right_wrist_angle_previous;
    Angle_diff.append([left_elbow_angle_diff,right_elbow_angle_diff,left_shoulder_angle_diff,right_shoulder_angle_diff,left_wrist_angle_diff,right_wrist_angle_diff])

def classifyPose(landmarks, output_image, display=False):
    global time_
    global Angle_previous
    global Angle_diff
    
    global left_elbow_angle
    global right_elbow_angle
    global left_shoulder_angle
    global right_shoulder_angle
    global left_wrist_angle
    global right_wrist_angle
    
    global left_elbow_angle_previous
    global right_elbow_angle_previous
    global left_shoulder_angle_previous
    global right_shoulder_angle_previous
    global left_wrist_angle_previous
    global right_wrist_angle_previous
    
    global left_elbow_angle_diff
    global right_elbow_angle_diff
    global left_shoulder_angle_diff
    global right_shoulder_angle_diff
    global left_wrist_angle_diff
    global right_wrist_angle_diff
    '''
    This function classifies yoga poses depending upon the angles of various body joints.
    Args:
        landmarks: A list of detected landmarks of the person whose pose needs to be classified.
        output_image: A image of the person with the detected pose landmarks drawn.
        display: A boolean value that is if set to true the function displays the resultant image with the pose label 
        written on it and returns nothing.
    Returns:
        output_image: The image with the detected pose landmarks drawn and pose label written.
        label: The classified pose label of the person in the output_image.
    '''
    
    # Initialize the label of the pose. It is not known at this stage.
    label = 'Unknown Pose'

    # Specify the color (Red) with which the label will be written on the image.
    color = (0, 0, 255)
    
    # Calculate the required angles.
    #----------------------------------------------------------------------------------------------------------------
    
    # Get the angle between the left shoulder, elbow and wrist points. 
    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    
    # Get the angle between the right shoulder, elbow and wrist points. 
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])   
    
    # Get the angle between the left elbow, shoulder and hip points. 
    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

    # Get the angle between the right hip, shoulder and elbow points. 
    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

    # Get the angle between the left hip, knee and ankle points. 
    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    # Get the angle between the right hip, knee and ankle points 
    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    
    LEFT_HAND = 33
    RIGHT_HAND = 33+21
    
    # Get the angle between the wrist, thumb_tip and pinky_tip
    left_wrist_angle = calculateAngle(landmarks[mp_pose.HandLandmark.THUMB_TIP.value+LEFT_HAND],
                                      landmarks[mp_pose.HandLandmark.WRIST.value+LEFT_HAND],
                                      landmarks[mp_pose.HandLandmark.PINKY_TIP.value+LEFT_HAND])

    right_wrist_angle = calculateAngle(landmarks[mp_pose.HandLandmark.THUMB_TIP.value+RIGHT_HAND],
                                      landmarks[mp_pose.HandLandmark.WRIST.value+RIGHT_HAND],
                                      landmarks[mp_pose.HandLandmark.PINKY_TIP.value+RIGHT_HAND])
    
    # print(mp_pose.HandLandmark.THUMB_TIP.value)
    # print(mp_pose.PoseLandmark.RIGHT_EYE_INNER.value)
    
    # Distance Point
    left_thumb_index_distance = calculateDistance(landmarks[mp_pose.HandLandmark.THUMB_TIP.value+LEFT_HAND],
                                      landmarks[mp_pose.HandLandmark.INDEX_FINGER_TIP.value+LEFT_HAND])

    right_thumb_index_distance = calculateDistance(landmarks[mp_pose.HandLandmark.THUMB_TIP.value+RIGHT_HAND],
                                      landmarks[mp_pose.HandLandmark.INDEX_FINGER_TIP.value+RIGHT_HAND])
    
    left_thumb_pinky_distance = calculateDistance(landmarks[mp_pose.HandLandmark.THUMB_TIP.value+LEFT_HAND],
                                      landmarks[mp_pose.HandLandmark.PINKY_TIP.value+LEFT_HAND])

    right_thumb_pinky_distance = calculateDistance(landmarks[mp_pose.HandLandmark.THUMB_TIP.value+RIGHT_HAND],
                                      landmarks[mp_pose.HandLandmark.PINKY_TIP.value+RIGHT_HAND])
    
    
    # Show Circlr Distance Point
    cirlcr1_posx ,cirlcr1_posy,_= landmarks[mp_pose.HandLandmark.THUMB_TIP.value+LEFT_HAND]
    cirlcr2_posx ,cirlcr2_posy,_= landmarks[mp_pose.HandLandmark.INDEX_FINGER_TIP.value+LEFT_HAND]
    circle_center = (int((cirlcr1_posx+cirlcr2_posx)*0.5),int((cirlcr1_posy+cirlcr2_posy)*0.5))
    cv2.circle(output_image,circle_center,int(left_thumb_index_distance/2.5),(0,0,0),-1)
    
    cirlcr1_posx ,cirlcr1_posy,_= landmarks[mp_pose.HandLandmark.THUMB_TIP.value+RIGHT_HAND]
    cirlcr2_posx ,cirlcr2_posy,_= landmarks[mp_pose.HandLandmark.INDEX_FINGER_TIP.value+RIGHT_HAND]
    circle_center = (int((cirlcr1_posx+cirlcr2_posx)*0.5),int((cirlcr1_posy+cirlcr2_posy)*0.5))
    cv2.circle(output_image,circle_center,int(right_thumb_index_distance/2.5),(0,0,0),-1)

    # Draw Data
    cv2.rectangle(output_image,(0,0),(380,480),(0,0,0),-1)
    text_posx = 20;
    text_step = 40
    cv2.putText(output_image, "L_elbow_angle : " + str("{:0.2f}".format(left_elbow_angle)), (10, text_posx),cv2.FONT_HERSHEY_PLAIN, 1.3, (255,255,255), 2)
    cv2.putText(output_image, "R_elbow_angle : " + str("{:0.2f}".format(right_elbow_angle)), (10, text_posx+text_step*1),cv2.FONT_HERSHEY_PLAIN, 1.3, (255,255,255), 2)
    cv2.putText(output_image, "L_shoulder_angle : " + str("{:0.2f}".format(left_shoulder_angle)), (10, text_posx+text_step*2),cv2.FONT_HERSHEY_PLAIN, 1.3, (255,255,255), 2)
    cv2.putText(output_image, "R_shoulder_angle : " + str("{:0.2f}".format(right_shoulder_angle)), (10, text_posx+text_step*3),cv2.FONT_HERSHEY_PLAIN, 1.3, (255,255,255), 2)
    cv2.putText(output_image, "L_wrist_angle : " + str("{:0.2f}".format(left_wrist_angle)), (10, text_posx+text_step*4),cv2.FONT_HERSHEY_PLAIN, 1.3, (255,255,255), 2)
    cv2.putText(output_image, "R_wrist_angle : " + str("{:0.2f}".format(right_wrist_angle)), (10, text_posx+text_step*5),cv2.FONT_HERSHEY_PLAIN, 1.3, (255,255,255), 2)
    
    cv2.putText(output_image, "L_thumb_index_distance : " + str("{:0.2f}".format(left_thumb_index_distance)), (10, text_posx+text_step*6),cv2.FONT_HERSHEY_PLAIN, 1.3, (0,255,0), 2)
    cv2.putText(output_image, "R_thumb_index_distance : " + str("{:0.2f}".format(right_thumb_index_distance)), (10, text_posx+text_step*7),cv2.FONT_HERSHEY_PLAIN, 1.3, (0,255,0), 2)
    cv2.putText(output_image, "L_thumb_pinky_distance : " + str("{:0.2f}".format(left_thumb_pinky_distance)), (10, text_posx+text_step*8),cv2.FONT_HERSHEY_PLAIN, 1.3, (0,255,0), 2)
    cv2.putText(output_image, "R_thumb_pinky_distance : " + str("{:0.2f}".format(right_thumb_pinky_distance)), (10, text_posx+text_step*9),cv2.FONT_HERSHEY_PLAIN, 1.3, (0,255,0), 2)
    
    #----------------------------------------------------------------------------------------------------------------
    
    # Update Diff Angle
    if(time.time() - time_ > 0.2):
        
        left_elbow_angle_diff = abs(left_elbow_angle - left_elbow_angle_previous);
        right_elbow_angle_diff = abs(right_elbow_angle - right_elbow_angle_previous);
        left_shoulder_angle_diff = abs(left_shoulder_angle - left_shoulder_angle_previous);
        right_shoulder_angle_diff = abs(right_shoulder_angle - right_shoulder_angle_previous);
        left_wrist_angle_diff = abs(left_wrist_angle - left_wrist_angle_previous);
        right_wrist_angle_diff = abs(right_wrist_angle - right_wrist_angle_previous);
        Angle_diff.append([left_elbow_angle_diff,right_elbow_angle_diff,left_shoulder_angle_diff,right_shoulder_angle_diff,left_wrist_angle_diff,right_wrist_angle_diff])
        
        # print(Angle_diff)
               
        left_elbow_angle_previous = left_elbow_angle;
        right_elbow_angle_previous = right_elbow_angle;
        left_shoulder_angle_previous = left_shoulder_angle;
        right_shoulder_angle_previous = right_shoulder_angle;
        left_wrist_angle_previous = left_wrist_angle;
        right_wrist_angle_previous = right_wrist_angle;
        
        Angle_previous = []
        Angle_diff = []
        time_ = time.time()
    
    
    # Check if it is the warrior II pose or the T pose.
    # As for both of them, both arms should be straight and shoulders should be at the specific angle.
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if the both arms are straight.
    if left_elbow_angle > 165 and left_elbow_angle < 195 and right_elbow_angle > 165 and right_elbow_angle < 195:

        # Check if shoulders are at the required angle.
        if left_shoulder_angle > 80 and left_shoulder_angle < 110 and right_shoulder_angle > 80 and right_shoulder_angle < 110:

    # Check if it is the warrior II pose.
    #----------------------------------------------------------------------------------------------------------------

            # Check if one leg is straight.
            if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:

                # Check if the other leg is bended at the required angle.
                if left_knee_angle > 90 and left_knee_angle < 120 or right_knee_angle > 90 and right_knee_angle < 120:

                    # Specify the label of the pose that is Warrior II pose.
                    label = 'Warrior II Pose' 
                        
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if it is the T pose.
    #----------------------------------------------------------------------------------------------------------------
    
            # Check if both legs are straight
            if left_knee_angle > 160 and left_knee_angle < 195 and right_knee_angle > 160 and right_knee_angle < 195:

                # Specify the label of the pose that is tree pose.
                label = 'T Pose'

    #----------------------------------------------------------------------------------------------------------------
    
    # Check if it is the tree pose.
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if one leg is straight
    if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:

        # Check if the other leg is bended at the required angle.
        if left_knee_angle > 315 and left_knee_angle < 335 or right_knee_angle > 25 and right_knee_angle < 45:

            # Specify the label of the pose that is tree pose.
            label = 'Tree Pose'
                
    #------===========================================================================================
    
    
    if(left_wrist_angle != 0 or right_wrist_angle != 0):
        if right_elbow_angle > 25 and right_elbow_angle < 110:
            if right_elbow_angle_diff > 10:
                print("bye right")
                a= 0
        if left_elbow_angle > 260 and left_elbow_angle < 340:
            if left_elbow_angle_diff > 10:
                print("bye left")
                a= 0
    

    #------===========================================================================================
    
    # Check if the pose is classified successfully
    if label != 'Unknown Pose':
        
        # Update the color (to green) with which the label will be written on the image.
        color = (0, 255, 0)  
    
    # Write the label on the output image. 
    
    # cv2.putText(output_image, label, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    
    # Update Varible
    
    
    # Check if the resultant image is specified to be displayed.
    if display:
    
        # Display the resultant image.
        plt.figure(figsize=[10,10])
        plt.title("Output Image");plt.axis('off');plt.imshow(output_image[:,:,::-1]);plt.show()
        
    else:
        
        # Return the output image and the classified label.
        return output_image, label
'''
############################################################3


