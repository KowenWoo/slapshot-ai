import mediapipe as mp
import cv2
import json
import os
import math
import numpy as np

# Paths for inference
<<<<<<< HEAD
ROOT = "C:/Users/kowen/OneDrive - University of Alberta/Projects/GoalGuru/local"
model_path = os.path.join(ROOT, "pose_landmarker_full.task")  # Ensure correct path
data_path = os.path.join(ROOT, "data")  # Path to folder to store data from running model
json_output_path = os.path.join(data_path, "poses")  # Folder for saving JSON data
data = {}

FOLDER = "C:/Users/kowen/OneDrive - University of Alberta/Projects/GoalGuru/UVic_data/P6"
VIDEOS = [os.path.join(FOLDER, file) for file in os.listdir(FOLDER) if file.endswith(".mp4")]

=======
ROOT = "C:/Users/kowen/OneDrive/GoalGuru/local"
model_path = os.path.join(ROOT, "pose_landmarker_full.task")  # Ensure correct path
video_path = "C:/Users/kowen/Videos/Captures/(104) How to Perform a Slap Shot - YouTube - Google Chrome 2024-06-26 19-30-46.mp4"
data_path = os.path.join(ROOT, "data")  # Path to folder to store data from running model
json_output_path = os.path.join(data_path, "pose_world_landmarks_data.json")  # Path for saving JSON data
data = {}

>>>>>>> b7c691eb6256d04a7ce7c84434160a716f4ebef2
# Write pose landmarks to dictionary
def write_landmarks_to_dict(landmarks, frame_number, data, delta_t, world=False):
    if frame_number not in data:
        data[frame_number] = {}

    frame_data = data[frame_number]
    landmark_type = "world_landmarks" if world else "pose_landmarks"

    if landmark_type not in frame_data:
        frame_data[landmark_type] = {}

    for idx, landmark in enumerate(landmarks):
<<<<<<< HEAD
        landmark_name = mp.solutions.pose.PoseLandmark(idx).name
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
            os.remove(file_path)
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        print(f"Data successfully written to {file_path}")

def main():
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose()

    os.makedirs(json_output_path, exist_ok=True)  # Ensure output directory exists

    for data_nbr, video in enumerate(VIDEOS, start=51):
        # Open the video file
        cap = cv2.VideoCapture(video)
        cv2.namedWindow('MediaPipe Pose', cv2.WINDOW_NORMAL)

        # Initialize VideoWriter object
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (frame_width, frame_height)
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter(f'output_videos/output{data_nbr}.avi', fourcc, fps, size)

        # Loop through frames and analyze pose
        frame_number = 0
        prev_frame_time = None
        video_data = {}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_frame_time = cap.get(cv2.CAP_PROP_POS_MSEC)
            delta_t = (current_frame_time - prev_frame_time) / 1000.0 if prev_frame_time else 0
            prev_frame_time = current_frame_time

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(frame_rgb)

            if result.pose_landmarks:
                mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                write_landmarks_to_dict(result.pose_landmarks.landmark, frame_number, video_data, delta_t, world=False)

            if result.pose_world_landmarks:
                write_landmarks_to_dict(result.pose_world_landmarks.landmark, frame_number, video_data, delta_t, world=True)

            out.write(frame)
            cv2.imshow('MediaPipe Pose', frame)

            if cv2.waitKey(1) == ord('q'):
                break

            frame_number += 1

        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # Save JSON for this video
        video_data["tot_frames"] = frame_number
        video_data["fps"] = fps
        json_file_path = os.path.join(json_output_path, f"pose_data_{data_nbr}.json")
        save_json(video_data, json_file_path)

if __name__ == "__main__":
    main()

=======
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
>>>>>>> b7c691eb6256d04a7ce7c84434160a716f4ebef2

