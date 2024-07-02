import mediapipe as mp
import cv2
import json
import os

# Paths for inference
ROOT = "C:/Users/kowen/OneDrive/Slapshot AI"
model_path = os.path.join(ROOT, "slapshot-ai/pose_landmarker_full.task") # Ensure correct path
video_path = "C:/Users/kowen/Videos/Captures/(104) How to Perform a Slap Shot - YouTube - Google Chrome 2024-06-26 19-30-46.mp4"

data = {}

def write_landmarks_to_dict(landmarks, frame_number, data):
    print(f"Landmark coordinates for frame {frame_number}:")
    
    # Initialize the frame's dictionary if it doesn't exist
    if frame_number not in data:
        data[frame_number] = {}

    # Loop through each landmark and add it to the frame's dictionary
    for idx, landmark in enumerate(landmarks):
        landmark_name = mp_pose.PoseLandmark(idx).name
        data[frame_number][landmark_name] = [landmark.x, landmark.y, landmark.z]
        print(f"{landmark_name}: (x: {landmark.x}, y: {landmark.y}, z: {landmark.z})")
    print("\n")

# Initialize MediaPipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Open the video file
cap = cv2.VideoCapture(video_path)
cv2.namedWindow('MediaPipe Pose', cv2.WINDOW_NORMAL)

# Initialize VideoWriter object with frame size
frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4)) 
size = (frame_width, frame_height) 
fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # Change parameter based on OS
out = cv2.VideoWriter('output.avi', fourcc, 20.0, size)

# Loop through frames and analyze pose
frame_number = 0
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

        # Add the landmark coordinates to the dictionary and print them
        write_landmarks_to_dict(result.pose_landmarks.landmark, frame_number, data)

    # Save frame
    out.write(frame)

    # Display the frame
    cv2.imshow('MediaPipe Pose', frame)

    # Exit if 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

    frame_number += 1

cap.release()
out.release()
cv2.destroyAllWindows()

# Save the dictionary to a JSON file
file_path = os.path.join(ROOT, 'data.json')
try:
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    print(f"Data successfully written to {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")
