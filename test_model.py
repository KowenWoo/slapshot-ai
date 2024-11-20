import mediapipe as mp
import cv2
import json
import os
import math
import numpy as np

# Paths for inference
ROOT = "C:/Users/kowen/OneDrive - University of Alberta/Projects/GoalGuru/local"
model_path = os.path.join(ROOT, "pose_landmarker_full.task")  # Ensure correct path
data_path = os.path.join(ROOT, "data")  # Path to folder to store data from running model
json_output_path = os.path.join(data_path, "poses")  # Folder for saving JSON data
data = {}

FOLDER = "C:/Users/kowen/OneDrive - University of Alberta/Projects/GoalGuru/UVic_data/P11/P11"
VIDEOS = [
    os.path.join(FOLDER, file) 
    for file in sorted(os.listdir(FOLDER)) 
    if file.endswith(".mp4")
]
print(VIDEOS)

# Write pose landmarks to dictionary
def write_landmarks_to_dict(landmarks, frame_number, data, delta_t, world=False):
    if frame_number not in data:
        data[frame_number] = {}

    frame_data = data[frame_number]
    landmark_type = "world_landmarks" if world else "pose_landmarks"

    if landmark_type not in frame_data:
        frame_data[landmark_type] = {}

    for idx, landmark in enumerate(landmarks):
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

    for data_nbr, video in enumerate(VIDEOS, start=102):
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
        json_file_path = os.path.join(json_output_path, f"pose_data_{data_nbr}.json")
        save_json(video_data, json_file_path)

if __name__ == "__main__":
    main()


