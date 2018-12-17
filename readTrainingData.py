# A function for creating the training data from the images and log file
import csv
import cv2
import numpy as np

def readTrainingData():
    images = []
    measurements = []
    with open('data/driving_log.csv') as csvfile:
        print('Reading in training data...')
        reader = iter(csv.reader(csvfile))
        next(reader) # skip first line
        printInc = 1000
        for i, line in enumerate(reader):
            if (i%printInc == 0) and (i != 0):
                print('Finished lines ' + str(i-printInc) + '-'+ str(i))
            sourcePath = line[0]
            fileName = sourcePath.split('/')[-1]
            currentPath = 'data/IMG/' + fileName
            images.append(cv2.imread(currentPath))
            measurements.append(float(line[3]))
            
        print('Finished reading in training data.')
        
    X_train = np.array(images) # shape: (8036, 160, 320, 3)
    y_train = np.array(measurements) # shape: (8036,)
    
    return X_train, y_train