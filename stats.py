import math
from cv2 import sqrt
import numpy as np
import os
import json

#coords path
ROOT = "C:/Users/kowen/OneDrive - University of Alberta/Projects/GoalGuru/local"
data_path = os.path.join(ROOT, "data")  # Path to folder to store data from running model
OUTPUT_PATH = os.path.join(ROOT, "data", "stats.json")
stats = {"hand_dist": [], "feet_dist": [], "hand_velocity": [], "rotation_sequence": []}
FOLDER = os.path.join(data_path, "poses")
POSES = [os.path.join(FOLDER, file) for file in os.listdir(FOLDER) if file.endswith(".json")]

def main():
    for data_nbr, coords in enumerate(POSES, start=1):
        with open(coords) as f:
            data = json.load(f)

        skip = []
        for i in range(data["tot_frames"]):

            if str(i) not in data.keys():
                skip.append(i)
                continue
            else:
                landmark = data[str(i)]["world_landmarks"]

            #hand distance calculations
            hand1 = np.array(landmark["LEFT_WRIST"])
            hand2 = np.array(landmark["RIGHT_WRIST"])
            shoulder_width = vector_distance(np.array(landmark["LEFT_SHOULDER"]), np.array(landmark["RIGHT_SHOULDER"]))
            stats["hand_dist"].append(hand_dist(hand1, hand2, shoulder_width))

            #hand velocity calculations
            if i == 0:
                continue
            else:
                if (i-1) in skip: #handle missing frames for angular displacement calculations
                    for j in range(i, skip[0], -1):
                        if j not in skip:
                            past_landmark = data[str(j)]["world_landmarks"]
                else:
                    past_landmark = data[str(i-1)]["world_landmarks"]

                # Previous frame vectors
                lh_0 = np.array(past_landmark["LEFT_WRIST"])
                rh_0 = np.array(past_landmark["RIGHT_WRIST"])

                # Current frame vectors
                lh_1 = np.array(landmark["LEFT_WRIST"])
                rh_1 = np.array(landmark["RIGHT_WRIST"])

                stats["hand_velocity"].append(vector_distance(lh_1, lh_0) + vector_distance(rh_1, rh_0))

            #feet distance calculations
            l_foot = np.array(landmark["LEFT_ANKLE"])
            r_foot = np.array(landmark["RIGHT_ANKLE"])
            stats["feet_dist"].append(feet_dist(l_foot, r_foot, shoulder_width))

            #rotational sequence calculations
            if i == 0:
                continue
            else:
                if (i-1) in skip: #handle missing frames for angular displacement calculations
                    for j in range(i, skip[0], -1):
                        if j not in skip:
                            past_landmark = data[str(j)]["world_landmarks"]
                else:
                    past_landmark = data[str(i-1)]["world_landmarks"]

                # Previous frame vectors
                sh_0 = (np.array(past_landmark["LEFT_SHOULDER"]) - np.array(past_landmark["RIGHT_SHOULDER"]))[:2]
                la_0 = (np.array(past_landmark["LEFT_WRIST"]) - np.array(past_landmark["LEFT_ELBOW"]))[:2]
                ra_0 = (np.array(past_landmark["RIGHT_WRIST"]) - np.array(past_landmark["RIGHT_ELBOW"]))[:2]
                h_0 = (np.array(past_landmark["LEFT_HIP"]) - np.array(past_landmark["RIGHT_HIP"]))[:2]

                # Current frame vectors
                sh_1 = (np.array(landmark["LEFT_SHOULDER"]) - np.array(landmark["RIGHT_SHOULDER"]))[:2]
                la_1 = (np.array(landmark["LEFT_WRIST"]) - np.array(landmark["LEFT_ELBOW"]))[:2]
                ra_1 = (np.array(landmark["RIGHT_WRIST"]) - np.array(landmark["RIGHT_ELBOW"]))[:2]
                h_1 = (np.array(landmark["LEFT_HIP"]) - np.array(landmark["RIGHT_HIP"]))[:2]
                #only using x and y coordinates to keep a defined plane for reference of rotation(transverse)
                
                stats["rotation_sequence"].append(rotation_sequence((sh_1, sh_0), (la_1, la_0), (ra_1, ra_0), (h_1, h_0)))

        save_json(stats, OUTPUT_PATH)

def vector_distance(vec1, vec2):
    '''
    calculate distance between two vectors
    '''
    return np.linalg.norm(vec1 - vec2)

def hand_dist(hand1, hand2, shoulder_width):
    '''
    calculate dist of left and right hands in proportion to shoulder width
    shoulder width is approx. 1/4 height
    '''
    return vector_distance(hand1, hand2) / shoulder_width

def feet_dist(foot1, foot2, shoulder_width):
    '''
    calculate distance between feet with proportion to shoulder width
    '''
    return vector_distance(foot1, foot2) / shoulder_width


def rotation_sequence(shoulders, l_arm, r_arm, hips):
    '''
    calculates angular displacement between 2 vectors
    '''
    # s_d = vector_displacement(shoulders[1], shoulders[0])
    # la_d = vector_displacement(l_arm[1], l_arm[0])
    # ra_d = vector_displacement(r_arm[1], r_arm[0])
    # h_d = vector_displacement(hips[1], hips[0])
    s_ra = rotation_angle(shoulders[1], shoulders[0])
    la_ra = rotation_angle(l_arm[1], l_arm[0])
    ra_ra = rotation_angle(r_arm[1], r_arm[0])
    h_ra = rotation_angle(hips[1], hips[0])
    return (s_ra, la_ra, ra_ra, h_ra)
    

# def vector_displacement(vec1, vec2):
#     return sqrt((vec1[0] - vec2[0])**2 + (vec1[1] - vec2[1])**2)

def rotation_angle(vec1, vec2):
    '''
    calculate angle between two vectors
    '''
    cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip to avoid domain errors
    return np.degrees(angle)  # Convert to degrees for easier interpretation

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

if __name__ == "__main__":
    main()
