import numpy as np

def flipTrainingData(X, y):
    X_flipped = []
    y_flipped = []
    for i in range(X.shape[0]):
        X_flipped.append(np.fliplr(X[i]))
        y_flipped.append(-y[i])

    X_result = np.vstack((X, np.array(X_flipped)))
    y_result = np.append(y, np.array(y_flipped))
    
    return X_result, y_result
