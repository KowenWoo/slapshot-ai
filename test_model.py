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


def extract_frames(video_path, output_dir, num_frames=10):
    """
    Extracts 10 frames from a video at 1/3, 1/2, and 2/3 marks.

    Parameters:
        video_path (str): Path to the input MP4 video file.
        output_dir (str): Directory to save the extracted frames.
        num_frames (int): Number of frames to extract at each mark.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video {video_path}")
        return

    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    # Calculate marks and step size
    marks = [total_frames // 3, total_frames // 2, (2 * total_frames) // 3]
    step_size = max(1, total_frames // (3 * num_frames))  # Distribute evenly

    print(f"Total frames: {total_frames}")
    print(f"Frame rate: {frame_rate}")
    print(f"Marks: {marks}, Step size: {step_size}")

    for mark in marks:
        for i in range(num_frames):
            frame_index = mark + (i - num_frames // 2) * step_size
            if frame_index < 0 or frame_index >= total_frames:
                continue  # Skip invalid indices
            
            # Set video to the frame index
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Unable to read frame {frame_index}")
                continue

            # Save the frame
            output_filename = os.path.join(output_dir, f"frame_{mark}_{i}.jpg")
            cv2.imwrite(output_filename, frame)
            print(f"Saved frame {frame_index} to {output_filename}")

    # Release the video capture object
    cap.release()
    print("Extraction complete.")

def main():
    for i in range(1, 12):
        if i > 6:
            FOLDER = f"C:/Users/kowen/OneDrive - University of Alberta/Projects/GoalGuru/UVic_data/P{i}/P{i}"
        else:
            FOLDER = f"C:/Users/kowen/OneDrive - University of Alberta/Projects/GoalGuru/UVic_data/P{i}"

        VIDEOS = [
            os.path.join(FOLDER, file)
            for file in sorted(os.listdir(FOLDER))
            if file.endswith(".mp4")
        ]

        # Initialize MediaPipe Pose
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        pose = mp_pose.Pose()

        os.makedirs(json_output_path, exist_ok=True)  # Ensure output directory exists
        if i == 1:
            start = 1
        else:
            start = (i - 1) * 11

        for data_nbr, video in enumerate(VIDEOS, start=start):
            # Extract frames from the video
            frame_output_dir = os.path.join(f"clipped_frames/P{i}", f"video_{data_nbr}")
            os.makedirs(frame_output_dir, exist_ok=True)
            extract_frames(video, frame_output_dir, num_frames=10)

            # Process extracted frames
            frame_files = sorted(
                [
                    os.path.join(frame_output_dir, frame)
                    for frame in os.listdir(frame_output_dir)
                    if frame.endswith(".jpg")
                ]
            )

            print(f"Processing frames for video: {video}")
            video_data = {}
            frame_number = 0

            for frame_path in frame_files:
                # Load the frame
                frame = cv2.imread(frame_path)

                if frame is None:
                    print(f"Error reading frame: {frame_path}")
                    continue

                # Process the frame with MediaPipe Pose
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(frame_rgb)

                # Save pose landmarks to JSON
                if result.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    write_landmarks_to_dict(result.pose_landmarks.landmark, frame_number, video_data, delta_t=0, world=False)

                if result.pose_world_landmarks:
                    write_landmarks_to_dict(result.pose_world_landmarks.landmark, frame_number, video_data, delta_t=0, world=True)

                # Show processed frame (optional)
                cv2.imshow('MediaPipe Pose', frame)
                if cv2.waitKey(1) == ord('q'):
                    break

                frame_number += 1

            # Save JSON for this video
            video_data["tot_frames"] = frame_number
            json_file_path = os.path.join(json_output_path, f"pose_data_{data_nbr}.json")
            save_json(video_data, json_file_path)

            print(f"Pose data saved for video {data_nbr}")

    # Cleanup
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


