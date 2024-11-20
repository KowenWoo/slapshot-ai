from scipy import stats
import numpy as np
import os
import json
 
KEYS = ['hand_dist', "feet_dist", "rotation_sequence"]

def main():
    data_path = os.path.join('data', 'stats.json')
    X = preprocess(data_path)
    Y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
                  0, 1, 1, 1, 0, 1, 1, 0, 0, 1,
                  1, 1, 0, 1, 1, 1, 0, 1, 1, 0,
                  1, 0, 1, 1, 1, 1, 1, 0, 1, 0,
                  1, 1, 1, 1, 0, 1, 1, 0, 1, 0,
                  1, 1, 1, 0, 1, 1, 0, 1, 1, 0,
                  1, 
                  1,
                  1,
                  1,
                  1,
                  ])
    

    '''
    X = data['X']
    y = data['y']

    # Fit the logistic regression model
    model = scipy.stats.binom.logit.fit(X, y)

    # Save the model
    with open('model.json', 'w') as f:
        json.dump(model, f)
    '''

def preprocess(data_path):
    '''
    Create 3d feature arrays from stats json file
    '''
    with open(data_path, 'r') as f:
        data = json.load(f)

    frames = min(len(data[KEYS[0]]), len(data[KEYS[1]]), len(data[KEYS[2]]) )

    X = []
    for i in range(frames):
        X[i] = np.array(data[KEYS[0]][i], data[KEYS[1]][i], data[KEYS[2]][i]) #hand_dist, feet_dist, rotation_sequence

    return X

if __name__ == '__main__':
    main()