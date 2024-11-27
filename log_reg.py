
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import numpy as np
import os
import json
from matplotlib import pyplot as plt

KEYS = ['hand_dist', "feet_dist", "hand_velocity_r", "hand_velocity_l", "rotation_sequence"]
RUN = False
CLASSIFIER = ["log_reg", "svm", "dec_tree", "rand_forest", "xgboost"]

def main():
    data_path = "C:/Users/kowen/OneDrive - University of Alberta/Projects/GoalGuru/local/data/stats"
    X_train, X_test = flatten_preprocess(data_path)
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
        # smote = SMOTE()
        # X_train, Y_train = smote.fit_resample(X_train, Y_train)
        # log_reg(X_train, Y_train, X_test, Y_test)
        # svm(X_train, Y_train, X_test, Y_test)
        # dec_tree(X_train, Y_train, X_test, Y_test)
        # rand_forest(X_train, Y_train, X_test, Y_test)
        mlp(X_train, Y_train, X_test, Y_test) #good training acc, below baseline test acc

    else:
        data_stats(Y_train, Y_test, X_train)


def log_reg(X_train, Y_train, X_test, Y_test):
    '''
    Logistic Regression
    '''
    model = LogisticRegression(class_weight='balanced')
    model.fit(X_train, Y_train)

    display_accuracy(model, X_train, Y_train, X_test, Y_test, "Logistic Regression")

    return 


def svm(X_train, Y_train, X_test, Y_test):
    '''
    Support Vector Machine
    '''
    model = SVC(kernel='rbf', class_weight='balanced')
    model.fit(X_train, Y_train)

    display_accuracy(model, X_train, Y_train, X_test, Y_test, "SVM")

    return


def dec_tree(X_train, Y_train, X_test, Y_test):
    '''
    Decision Tree
    '''
    model = DecisionTreeClassifier(class_weight='balanced')
    model.fit(X_train, Y_train)

    display_accuracy(model, X_train, Y_train, X_test, Y_test, "Decision Tree")

    return


def rand_forest(X_train, Y_train, X_test, Y_test):
    '''
    Random Forest
    '''
    model = RandomForestClassifier(class_weight='balanced')
    model.fit(X_train, Y_train)

    display_accuracy(model, X_train, Y_train, X_test, Y_test, "Random Forest")

    return


def mlp(X_train, Y_train, X_test, Y_test):
    '''
    Multi-Layer Perceptron
    '''
    EPOCHS = 25
    model = MLPClassifier(hidden_layer_sizes=(50,), activation="relu", solver="adam", learning_rate_init=0.01,
                           learning_rate="adaptive", max_iter=EPOCHS, shuffle=True, early_stopping=True, validation_fraction=0.1
                           )
    
    train_loss = []
    test_loss = []
    for epoch in range(EPOCHS):
        model.fit(X_train, Y_train)  
        # Record training loss
        train_loss.append(model.loss_)
        # Calculate test loss
        y_test_prob = model.predict_proba(X_test)
        test_loss.append(log_loss(Y_test, y_test_prob))

    display_accuracy(model, X_train, Y_train, X_test, Y_test, "MLP", train_loss, test_loss)

    return


# def voting(X_train, Y_train, X_test, Y_test):
#     '''
#     Voting Classifier
#     '''
#     model1 = LogisticRegression()
#     model2 = RandomForestClassifier()
#     model3 = SVC(probability=True)

#     model = VotingClassifier(estimators=[
#         ('lr', model1), ('rf', model2), ('svc', model3)], voting='soft')
#     model.fit(X_train, Y_train)

#     display_accuracy(model, X_train, Y_train, X_test, Y_test, "Voting Classifier")

#     return


def display_accuracy(model, X_train, Y_train, X_test, Y_test, model_name, train_loss=None, test_loss=None):
    '''
    Display accuracy for each classifier
    '''
    train_accuracy = model.score(X_train, Y_train)
    print(f"Training Accuracy ({model_name}): {train_accuracy}")

    # Run inference on test set
    Y_pred = model.predict(X_test)
    accuracy = np.mean(Y_pred == Y_test)
    print(f"Test Accuracy ({model_name}): {accuracy:.2f}")

    if model_name == "MLP":
        plt.plot(train_loss, label="Train Loss")
        plt.plot(test_loss, label="Test Loss")
        plt.title("Train vs Test Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.show()

    return


def flatten_preprocess(data_path):
    stats = [os.path.join(data_path, file) for file in os.listdir(data_path) if file.endswith(".json")]
    FIXED = 20 #frames to process for each file

    X_train = []
    for i in range(70):
        with open(stats[i]) as f:
            data = json.load(f)

        clip = []
        for i in KEYS:
            if i == "rotation_sequence":
                for j in range(FIXED):
                    a,b,c,d = data[i][j]
                    clip.append(a)
                    clip.append(b)
                    clip.append(c)
                    clip.append(d)
            else:
                clip.append(data[i][:FIXED])
        
        X_train.append(clip)

    X_test = []
    for i in range(70, 110):
        with open(stats[i]) as f:
            data = json.load(f)

        clip = []
        for i in KEYS:
            if i == "rotation_sequence":
                for j in range(FIXED):
                    a,b,c,d = data[i][j]
                    clip.append(a)
                    clip.append(b)
                    clip.append(c)
                    clip.append(d)
            else:
                clip.append(data[i][:FIXED])
        
        X_test.append(clip)

    print(len(X_train), len(X_test))
    print(X_train)

    return np.array(X_train), np.array(X_test)


def agg_preprocess(data_path):
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
    '''
    Baseline accuracy = 65.766%
    '''
    # Display class distribution and correlation with Y_train
    Y = np.concatenate((Y_train, Y_test))
    unique, counts = np.unique(Y, return_counts=True)
    print("Class Distribution in Y_train:", dict(zip(unique, counts)))

    # Calculate correlation between features and Y_train
    for i, key in enumerate(KEYS):
        corr = np.corrcoef(X_train[:, i], Y_train)[0, 1]
        print(f"Correlation between {key} and Y_train: {corr}")

if __name__ == "__main__":
    main()