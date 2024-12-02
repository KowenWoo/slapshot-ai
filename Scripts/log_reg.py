
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
import numpy as np
import os
import json
from matplotlib import pyplot as plt

KEYS = ['hand_dist', "feet_dist", "hand_velocity_r", "hand_velocity_l", "rotation_sequence"]
RUN = True
CLASSIFIER = ["log_reg", "svm", "dec_tree", "rand_forest", "mlp"]
ACCURACIES = []

def main():
    data_path = "C:/Users/kowen/OneDrive - University of Alberta/Projects/GoalGuru/local/data/stats"
    X_train, X_test = agg_preprocess(data_path)
    Y_train = np.array([  
        1, 1, 1, 1, 1, 1, 1, 1, 0, 1, #got rid of video 1, had too many frames for some reason
        0, 1, 1, 1, 0, 1, 1, 0, 0, 1,
        1, 1, 0, 1, 1, 1, 0, 1, 1, 0,
        1, 0, 1, 1, 1, 1, 1, 0, 1, 0,
        1, 1, 1, 1, 0, 1, 1, 0, 1, 0,
        1, 1, 1, 0, 1, 1, 0, 1, 1, 0,
        1, 1, 0, 1, 1, 0, 1, 0 ,1, 0
    ])
    Y_test = np.array([
        1, 1, 0, 1, 0, 1, 0, 0, 1, 0,
        1, 0, 1, 1, 0, 1, 0, 1, 0, 0, #error in processing video for p9
        0, 1, 0, 1, 1, 0, 1, 1, 1, 0,
        0, 1, 0, 1, 0, 1, 1, 0, 1, 1
    ])

    if RUN:
        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Handle class imbalance with SMOTE
        # smote = SMOTE()
        # X_train, Y_train = smote.fit_resample(X_train, Y_train)
        log_reg(X_train, Y_train, X_test, Y_test)
        svm(X_train, Y_train, X_test, Y_test)
        dec_tree(X_train, Y_train, X_test, Y_test)
        rand_forest(X_train, Y_train, X_test, Y_test)
        mlp(X_train, Y_train, X_test, Y_test) #good training acc, below baseline test acc
        # lstm_with_l2(X_train, Y_train, X_test, Y_test)

        #bar graph of accuracies
        plt.bar(CLASSIFIER, ACCURACIES)
        plt.title("Classifier Accuracies")
        plt.xlabel("Classifier")
        plt.ylabel("Accuracy")
        plt.grid()
        plt.show()

    # else:
        # data_stats(Y_train, Y_test, X_train)
        # X_stats(X_train)


def lstm_with_l2(X_train, Y_train, X_test, Y_test, sequence_length=20, features=8, l2_lambda=0.01):
    '''
    LSTM with L2 Regularization
    '''
    model = Sequential([
        LSTM(64, input_shape=(sequence_length, features), 
             return_sequences=False, 
             kernel_regularizer=l2(l2_lambda)),
        Dropout(0.1),
        Dense(32, activation='relu', kernel_regularizer=l2(l2_lambda)),
        Dense(1, activation='sigmoid', kernel_regularizer=l2(l2_lambda))  # Binary classification
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Training
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=15, batch_size=32)

    # Visualize loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

    return model

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
    EPOCHS = 20
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
    ACCURACIES.append(accuracy)

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
    stats = []
    stats = [os.path.join(data_path, file) for file in sorted(os.listdir(data_path)) if file.endswith(".json")]
    FIXED = 20 #frames to process for each file

    X_train = []
    for i in range(70):
        with open(stats[i]) as f:
            data = json.load(f)
            f1 = data[KEYS[0]]
            f2 = data[KEYS[1]]
            f3 = data[KEYS[2]]
            f4 = data[KEYS[3]]
            f5 = data[KEYS[4]]

        clip = []
        for i in range(FIXED):
            features = [f1[i], f2[i], f3[i], f4[i], *f5[i]]
            clip.append(features)

        X_train.append(clip)

    X_test = []
    for i in range(70, 110):
        with open(stats[i]) as f:
            data = json.load(f)
            f1 = data[KEYS[0]]
            f2 = data[KEYS[1]]
            f3 = data[KEYS[2]]
            f4 = data[KEYS[3]]
            f5 = data[KEYS[4]]

        clip = []
        for i in range(FIXED):
            features = [f1[i], f2[i], f3[i], f4[i], *f5[i]]
            clip.append(features)
        X_test.append(clip)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    print(X_train.shape, X_test.shape)
    return  X_train, X_test


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
    for i in range(70, len(stats)-5):  # ~30% test
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

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    print(X_train.shape, X_test.shape)
    return  X_train, X_test


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

def X_stats(X_train):
    '''
    Display statistics for each feature in X_train
    '''
    for i, key in enumerate(KEYS):
        feature = X_train[:, i]
        print(f"Feature: {key}")
        print(f"Mean: {np.mean(feature):.2f}")
        print(f"Std Dev: {np.std(feature):.2f}")
        print(f"Min: {np.min(feature):.2f}")
        print(f"Max: {np.max(feature):.2f}")
        print()

    return

if __name__ == "__main__":
    main()
