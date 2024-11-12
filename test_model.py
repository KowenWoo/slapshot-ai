import mediapipe as mp
import cv2
import json
import os
import math
import numpy as np

# Paths for inference
ROOT = "C:/Users/kowen/OneDrive/GoalGuru/local"
model_path = os.path.join(ROOT, "pose_landmarker_full.task")  # Ensure correct path
video_path = "C:/Users/kowen/Videos/Captures/(104) How to Perform a Slap Shot - YouTube - Google Chrome 2024-06-26 19-30-46.mp4"
data_path = os.path.join(ROOT, "data")  # Path to folder to store data from running model
json_output_path = os.path.join(data_path, "pose_world_landmarks_data.json")  # Path for saving JSON data
data = {}

# Write pose landmarks to dictionary
def write_landmarks_to_dict(landmarks, frame_number, data, delta_t, world=False):
    if frame_number not in data:
        data[frame_number] = {}

    frame_data = data[frame_number]
    landmark_type = "world_landmarks" if world else "pose_landmarks"

    if landmark_type not in frame_data:
        frame_data[landmark_type] = {}

    for idx, landmark in enumerate(landmarks):
        landmark_name = mp_pose.PoseLandmark(idx).name
        frame_data[landmark_type][landmark_name] = [landmark.x, landmark.y, landmark.z]

# Save the dictionary to a JSON file
def save_json(data, file_path):
    try:
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        print(f"Data successfully written to {file_path}")
    except Exception as e:
        print(f"Failed to write data: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)  # Overwrite if the file already exists
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        print(f"Data successfully written to {file_path}")

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
prev_frame_time = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate the time difference between the current frame and the previous frame
    current_frame_time = cap.get(cv2.CAP_PROP_POS_MSEC)
    if prev_frame_time is not None:
        delta_t = (current_frame_time - prev_frame_time) / 1000.0
    else:
        delta_t = 0
    prev_frame_time = current_frame_time

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert the frame to RGB
    result = pose.process(frame_rgb)  # Process the frame with MediaPipe Pose

    # Draw the pose landmarks on the frame
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Add 2D pose landmarks (screen space) to the dictionary
        #write_landmarks_to_dict(result.pose_landmarks.landmark, frame_number, data, delta_t, world=False)

    # Add world landmarks (3D space) to the dictionary
    if result.pose_world_landmarks:
        write_landmarks_to_dict(result.pose_world_landmarks.landmark, frame_number, data, delta_t, world=True)

    out.write(frame)  # Save frame
    cv2.imshow('MediaPipe Pose', frame)  # Display the frame
    if cv2.waitKey(1) == ord('q'):  # Exit if 'q' key is pressed
        break

    frame_number += 1

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

# Save collected landmark data to JSON file
data["tot_frames"] = frame_number
save_json(data, json_output_path)

