import math
from cv2 import sqrt
from matplotlib import pyplot as plt
import numpy as np
import os
import json

#coords path
ROOT = "C:/Users/kowen/OneDrive - University of Alberta/Projects/GoalGuru/local"
data_path = os.path.join(ROOT, "data")  # Path to folder to store data from running model
OUTPUT_PATH = os.path.join(data_path, "stats")
stats = {"hand_dist": [], "feet_dist": [], "hand_velocity": [], "rotation_sequence": []}
FOLDER = os.path.join(data_path, "poses")
POSES = [os.path.join(FOLDER, file) for file in os.listdir(FOLDER) if file.endswith(".json")]
FPS = 30

def main():
    for data_nbr, coords in enumerate(POSES, start=1):
        with open(coords) as f:
            data = json.load(f)

        skip = []
        stats = {"hand_dist": [], "hand_velocity_l": [], "hand_velocity_r": [], "feet_dist": [], "rotation_sequence": []}
        
        cnt = 0
        for i in data.keys():

            landmark = data[i]

            # Hand distance calculations
            hand1 = np.array(landmark["LEFT_WRIST"])
            hand2 = np.array(landmark["RIGHT_WRIST"])
            shoulder_width = vector_distance(
                np.array(landmark["LEFT_SHOULDER"]), np.array(landmark["RIGHT_SHOULDER"])
            )
            stats["hand_dist"].append(hand_dist(hand1, hand2))

            # Hand velocity calculations
            if cnt > 0:
                # Previous frame vectors
                lh_0 = np.array(past_landmark["LEFT_WRIST"])
                rh_0 = np.array(past_landmark["RIGHT_WRIST"])

                # Current frame vectors
                lh_1 = np.array(landmark["LEFT_WRIST"])
                rh_1 = np.array(landmark["RIGHT_WRIST"])

                # Calculate velocity
                stats["hand_velocity_l"].append(hand_velocity(lh_1, lh_0, 1 / FPS))
                stats["hand_velocity_r"].append(hand_velocity(rh_1, rh_0, 1 / FPS))

            # Feet distance calculations
            l_foot = np.array(landmark["LEFT_ANKLE"])
            r_foot = np.array(landmark["RIGHT_ANKLE"])
            stats["feet_dist"].append(feet_dist(l_foot, r_foot, shoulder_width))

            # Rotational sequence calculations
            if cnt > 0:
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

                # Calculate rotational sequence
                stats["rotation_sequence"].append(
                    rotation_sequence((sh_1, sh_0), (la_1, la_0), (ra_1, ra_0), (h_1, h_0))
                )

            past_landmark = data[i]
            cnt += 1

        # Save JSON
        save_json(stats, os.path.join(OUTPUT_PATH, f"{data_nbr}.json"))

def vector_distance(vec1, vec2):
    """Calculate distance between two vectors."""
    return np.linalg.norm(vec1 - vec2)


def hand_dist(hand1, hand2):
    """Calculate distance between hands."""
    return vector_distance(hand1, hand2)


def hand_velocity(hand1, hand2, delta_t):
    """Calculate velocity of hands."""
    return vector_distance(hand1, hand2) / delta_t


def feet_dist(foot1, foot2, shoulder_width):
    """Calculate distance between feet relative to shoulder width."""
    return vector_distance(foot1, foot2) / shoulder_width


def rotation_sequence(shoulders, l_arm, r_arm, hips):
    """Calculate angular displacement between vectors."""
    s_ra = rotation_angle(shoulders[1], shoulders[0])
    la_ra = rotation_angle(l_arm[1], l_arm[0])
    ra_ra = rotation_angle(r_arm[1], r_arm[0])
    h_ra = rotation_angle(hips[1], hips[0])
    return (s_ra, la_ra, ra_ra, h_ra)


def rotation_angle(vec1, vec2):
    """Calculate angle between two vectors in degrees."""
    cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip to avoid domain errors
    return np.degrees(angle)


def save_json(data, file_path):
    """Save data to a JSON file."""
    try:
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        print(f"Data successfully written to {file_path}")
    except Exception as e:
        print(f"Failed to write data: {e}")


def clip_stats():
    '''
    Clip frames of stats files to include 10 frames at:
        1/3 mark
        2/3 mark
        1/2 mark
    '''
    stats = os.path.join(ROOT, "data/stats")
    output_path = os.path.join(ROOT, "data/clipped_stats")
    KEYS = ["hand_dist", "feet_dist", "hand_velocity_r", "hand_velocity_l", "rotation_sequence"]

    min_frames = 10000
    max_frames = 0
    tot_frames = []
    # with open(os.path.join(ROOT, "data/stats/1.json")) as f:
    #     data = json.load(f)
    #     for i in KEYS:
    #         frames = len(data[i])
    #         tot_frames.append(frames)
    #     print(tot_frames)
    for file in os.listdir(stats):
        with open(os.path.join(stats, file)) as f:
            data = json.load(f)
        
        frames = min([len(data[key]) for key in KEYS])
        if frames < min_frames:
            min_frames = frames
            min_stat = file
        if frames > max_frames:
            max_frames = frames
        tot_frames.append(frames)

        for i in range(frames):
            if i == math.ceil(frames/3) or i == math.ceil(2*frames/3) or i == math.ceil(frames/2):
                # Clip stats
                for key in data.keys():
                    data[key] = data[key][:i+1]

        save_json(data, os.path.join(output_path, file))
    print(f"Min frames{min_stat}: {min_frames}")
    print(f"Max frames: {max_frames}")
    plt.hist(tot_frames, bins=20)
    plt.show()

def check_pose_frames():
    '''
    Check number of frames in each file
    '''
    poses = os.path.join(ROOT, "data/poses")
    tot_frames = []
    for file in os.listdir(poses):
        with open(os.path.join(poses, file)) as f:
            data = json.load(f)
        tot_frames.append(len(data.keys()))
        if len(data.keys()) > 30:
            print(file)
    
    print(min(tot_frames), max(tot_frames))
    plt.hist(tot_frames)
    plt.show()

def check_stats_frames():
    '''
    Check number of frames in each file
    '''
    stats = os.path.join(ROOT, "data/stats")
    tot_frames = []
    min_frames = []
    for file in os.listdir(stats):
        if file.endswith(".json"):
            with open(os.path.join(stats, file)) as f:
                data = json.load(f)
            l_hand = len(data["hand_dist"])
            l_foot = len(data["feet_dist"])
            r_hand = len(data["hand_velocity_r"])
            r_foot = len(data["hand_velocity_l"])
            tot_frames.append(np.average([l_hand, l_foot, r_hand, r_foot]))
            min_frames.append(min([l_hand, l_foot, r_hand, r_foot]))
            if np.average([l_hand, l_foot, r_hand, r_foot]) < 20 or np.average([l_hand, l_foot, r_hand, r_foot]) > 30:
                os.remove(os.path.join(stats, file))
    
    print(tot_frames)
    print(min(min_frames))
    plt.hist(tot_frames)
    plt.show()   

import os
import json
import matplotlib.pyplot as plt

def stat_over_time():
    '''
    Plot stats over time for each key in the KEYS list, and label rotational sequence for specific body parts.
    ROOT: base directory for data (where "data/stats" is located)
    '''
    stats = os.path.join(ROOT, "data/stats")
    KEYS = ["hand_dist", "feet_dist", "hand_velocity_r", "hand_velocity_l", "rotation_sequence"]
    
    with open("C:/Users/kowen/OneDrive - University of Alberta/Projects/GoalGuru/local/data/stats/test.json", "r") as f:
        data = json.load(f)
    
    # Plotting other stats first
    for key in KEYS:
        if key != "rotation_sequence":
            plt.plot(data[key])
            plt.title(f'{key} over time')
            plt.xlabel('Time (frames)')
            plt.ylabel(key)
            plt.show()
        else:
            # Handle rotation sequence specifically
            rotation_data = data[key]  # Assuming rotation_sequence is a list of tuples or lists
            # Assuming the rotation_sequence is structured as [shoulders, left_arm, right_arm, hips] for each frame
            # Each item in the sequence could be a tuple like (x, y, z, w) for the rotation

            # Separate each body part (shoulders, l arm, r arm, hips)
            shoulders = [frame[0] for frame in rotation_data]  # Assuming shoulders are in the first position
            l_arm = [frame[1] for frame in rotation_data]  # Assuming left arm is in the second position
            r_arm = [frame[2] for frame in rotation_data]  # Assuming right arm is in the third position
            hips = [frame[3] for frame in rotation_data]  # Assuming hips are in the fourth position
            
            # Plot each part with labels
            plt.plot(shoulders, label='Shoulders')
            plt.plot(l_arm, label='Left Arm')
            plt.plot(r_arm, label='Right Arm')
            plt.plot(hips, label='Hips')

            plt.title('Rotational Sequence Over Time')
            plt.xlabel('Time (frames)')
            plt.ylabel('Rotation')
            plt.legend()
            plt.show()


# clip_stats()
# check_stats_frames()
# check_pose_frames()
stat_over_time()
# if __name__ == "__main__":
#     main()
