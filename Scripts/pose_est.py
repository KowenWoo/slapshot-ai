import os
import json
import mediapipe as mp
import cv2

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils

# Function to write pose landmarks to dictionary
def write_landmarks_to_dict(landmarks, image_name, data_dict):
    """
    Write pose landmarks to a dictionary.
    
    Parameters:
        landmarks: Detected landmarks from MediaPipe.
        image_name (str): Name of the image file.
        data_dict (dict): Dictionary to store the results.
    """
    data_dict[image_name] = {}
    for idx, landmark in enumerate(landmarks):
        landmark_name = mp_pose.PoseLandmark(idx).name
        data_dict[image_name][landmark_name] = [landmark.x,landmark.y,landmark.z]


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

def make_raw_string(input_string):
    return r"{}".format(input_string)


# Main function
def main():
    data = "C:/Users/kowen/OneDrive - University of Alberta/Projects/GoalGuru/local/clipped_frames"  # Update with your folder path
    OUTPUT_JSON_FOLDER = "C:/Users/kowen/OneDrive - University of Alberta/Projects/GoalGuru/local/data/poses"

    for player_folder in sorted(os.listdir(data)): #loop through player folders
        for video_folder in sorted(os.listdir(os.path.join(data, player_folder))): #loop through video folders
            pose_data = {}
            for img in sorted(os.listdir(os.path.join(data, player_folder, video_folder))): #loop through images
                if not img.endswith(".jpg"):
                    continue
                
                img_path = os.path.join(data, player_folder, video_folder, img)
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Error loading image: {img_path}")
                    continue

                
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Read image and convert to RGB
                # mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                
                # Perform pose estimation
                results = pose.process(image_rgb)
                
                if results.pose_world_landmarks:
                    # Write real-world landmarks to the dictionary
                    write_landmarks_to_dict(results.pose_world_landmarks.landmark, img, pose_data)
                else:
                    print(f"No pose landmarks detected in {img}")

                

            # Save results to JSON file
            save_json(pose_data, os.path.join(OUTPUT_JSON_FOLDER, f"{player_folder}_{video_folder}.json"))

if __name__ == "__main__":
    main()