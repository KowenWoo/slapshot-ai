import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2 
import numpy as np
import csv
# paths for inference - CHANGE TO RUN
ROOT = "C:/Users/kowen/OneDrive/Slapshot AI" 
model_path = ROOT + "slapshot-ai/pose_landmarker_full.task" # Download pose_landmarker (full) from mediapipe
video_path = "C:/Users/kowen/Videos/Captures/(104) How to Perform a Slap Shot - YouTube - Google Chrome 2024-06-26 19-30-46.mp4"
output_csv = ROOT

def write_landmarks_to_csv(landmarks, frame_number, csv_data):
    print(f"Landmark coordinates for frame {frame_number}:")
    for idx, landmark in enumerate(landmarks):
        print(f"{mp_pose.PoseLandmark(idx).name}: (x: {landmark.x}, y: {landmark.y}, z: {landmark.z})")
        csv_data.append([frame_number, mp_pose.PoseLandmark(idx).name, landmark.x, landmark.y, landmark.z])
    print("\n")

# Initialize MediaPipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Open the video file
cap = cv2.VideoCapture(video_path)
cv2.namedWindow('MediaPipe Pose', cv2.WINDOW_NORMAL)

# initialize videowriter object with frame size
frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4)) 
size = (frame_width, frame_height) 
fourcc = cv2.VideoWriter_fourcc(*'DIVX') #CHANGE PARAM BASED ON OS
out = cv2.VideoWriter('output.avi', fourcc, 20.0, size)

# loop through frames and analyze pose
frame_number = 0
csv_data = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Pose
    result = pose.process(frame_rgb)

    # Draw the pose landmarks on the frame
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Add the landmark coordinates to the list and print them
        write_landmarks_to_csv(result.pose_landmarks.landmark, frame_number, csv_data)

    # save frame
    out.write(frame)
    # Display the frame
    cv2.imshow('MediaPipe Pose', frame)

    # Exit if 'q' keypyt or 
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()