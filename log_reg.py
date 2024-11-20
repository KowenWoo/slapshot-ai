from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import numpy as np
import os
import json

import test

KEYS = ['hand_dist', "feet_dist", "hand_velocity_r", "hand_velocity_l", "rotation_sequence"]
RUN = False

def main():
    data_path = "C:/Users/kowen/OneDrive - University of Alberta/Projects/GoalGuru/local/data/stats"
    X_train, X_test = preprocess(data_path)
    Y_train = np.array([
        1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
        0, 1, 1, 1, 0, 1, 1, 0, 0, 1,
        1, 1, 0, 1, 1, 1, 0, 1, 1, 0,
        1, 0, 1, 1, 1, 1, 1, 0, 1, 0,
        1, 1, 1, 1, 0, 1, 1, 0, 1, 0,
        1, 1, 1, 0, 1, 1, 0, 1, 1, 0,
        1, 1, 0, 1, 1, 0, 1, 0 ,1, 0
    ])
    Y_test = np.array([
        1, 1, 0, 1, 0, 1, 0, 0, 1, 0,
        1, 1, 0, 1, 1, 0, 1, 0, 1, 0,
        0, 1, 0, 1, 1, 0, 1, 1, 1, 0,
        0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1
    ])

    if RUN:
        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Handle class imbalance with SMOTE
        smote = SMOTE()
        X_train, Y_train = smote.fit_resample(X_train, Y_train)

        # Fit the logistic regression model
        model = LogisticRegression(class_weight='balanced')
        model.fit(X_train, Y_train)

        # Evaluate the model on the training set
        train_accuracy = model.score(X_train, Y_train)
        print("Training Accuracy:", train_accuracy)

        # Save the model coefficients
        model_data = {"coef_": model.coef_.tolist(), "intercept_": model.intercept_.tolist()}
        with open('model.json', 'w') as f:
            json.dump(model_data, f)
        print("Model saved successfully.")

        # Run inference on test set
        Y_pred = model.predict(X_test)

        # Evaluate the model
        accuracy = np.mean(Y_pred == Y_test)
        print(f"Test Accuracy: {accuracy:.2f}")

        # Optionally, display predictions
        print("Predictions:", Y_pred)
        print("Ground Truth:", Y_test)

    else:
        data_stats(Y_train, Y_test, X_train)

'''
def preprocess(data_path):
    stats = [os.path.join(data_path, file) for file in os.listdir(data_path) if file.endswith(".json")]
    print(len(stats))

    X_train = []
    for i in range(70):  # ~70% train
        with open(stats[i]) as f:
            data = json.load(f)

        frames = min(
            len(data[KEYS[0]]),
            len(data[KEYS[1]]),
            len(data[KEYS[2]]),
            len(data[KEYS[3]]),
            len(data[KEYS[4]])
        )

        for j in range(frames):
            features = [
                data[KEYS[0]][j],
                data[KEYS[1]][j],
                data[KEYS[2]][j],
                data[KEYS[3]][j]
            ]
            # Append the rotational sequence as a tuple
            features.extend(data[KEYS[4]][j])  # Unpack tuple values
            X_train.append(features)

    X_test = []
    for i in range(70, len(stats)):  # ~30% test
        with open(stats[i]) as f:
            data = json.load(f)

        frames = min(
            len(data[KEYS[0]]),
            len(data[KEYS[1]]),
            len(data[KEYS[2]]),
            len(data[KEYS[3]]),
            len(data[KEYS[4]])
        )

        for j in range(frames):
            features = [
                data[KEYS[0]][j],
                data[KEYS[1]][j],
                data[KEYS[2]][j],
                data[KEYS[3]][j]
            ]
            # Append the rotational sequence as a tuple
            features.extend(data[KEYS[4]][j])  # Unpack tuple values
            X_test.append(features)

    return np.array(X_train), np.array(X_test)
'''

def preprocess(data_path):
    '''
    Create 3D feature arrays from stats JSON files.
    Aggregates features across frames for each file.
    '''
    stats = [os.path.join(data_path, file) for file in os.listdir(data_path) if file.endswith(".json")]

    X_train = []
    for i in range(70):  # ~70% train
        with open(stats[i]) as f:
            data = json.load(f)

        # Aggregate features across all frames (e.g., average)
        frames = min(
            len(data[KEYS[0]]),
            len(data[KEYS[1]]),
            len(data[KEYS[2]]),
            len(data[KEYS[3]]),
            len(data[KEYS[4]])
        )
        aggregated_features = []
        for j in range(frames):
            frame_features = [
                data[KEYS[0]][j],
                data[KEYS[1]][j],
                data[KEYS[2]][j],
                data[KEYS[3]][j],
                *data[KEYS[4]][j]  # Unpack tuple
            ]
            aggregated_features.append(frame_features)

        # Average features across frames
        aggregated_features = np.mean(aggregated_features, axis=0)
        X_train.append(aggregated_features)

    X_test = []
    for i in range(70, len(stats)-1):  # ~30% test
        with open(stats[i]) as f:
            data = json.load(f)

        frames = min(
            len(data[KEYS[0]]),
            len(data[KEYS[1]]),
            len(data[KEYS[2]]),
            len(data[KEYS[3]]),
            len(data[KEYS[4]])
        )
        aggregated_features = []
        for j in range(frames):
            frame_features = [
                data[KEYS[0]][j],
                data[KEYS[1]][j],
                data[KEYS[2]][j],
                data[KEYS[3]][j],
                *data[KEYS[4]][j]  # Unpack tuple
            ]
            aggregated_features.append(frame_features)

        # Average features across frames
        aggregated_features = np.mean(aggregated_features, axis=0)
        X_test.append(aggregated_features)

    return np.array(X_train), np.array(X_test)

def data_stats(Y_train, Y_test, X_train):
    # Display class distribution and correlation with Y_train
    unique, counts = np.unique(Y_train, return_counts=True)
    print("Class Distribution in Y_train:", dict(zip(unique, counts)))

    # Calculate correlation between features and Y_train
    for i, key in enumerate(KEYS):
        corr = np.corrcoef(X_train[:, i], Y_train)[0, 1]
        print(f"Correlation between {key} and Y_train: {corr}")



if __name__ == "__main__":
    main()