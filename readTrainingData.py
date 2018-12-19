# A function for creating the training data from the images and log file
import csv
import cv2
import numpy as np

def readTrainingData():
    images = []
    steeringAngles = []
    with open('data/driving_log.csv') as csvfile:
        print('Reading in training data...')
        reader = iter(csv.reader(csvfile))
        next(reader) # skip first line
        printInc = 1000
        for i, line in enumerate(reader):
            # print progress
            if (i%printInc == 0) and (i != 0):
                print('Finished lines ' + str(i-printInc) + '-'+ str(i))
                
            # add images
            imageCenter = cv2.imread('data/IMG/' + line[0].split('/')[-1])
            imageLeft = cv2.imread('data/IMG/' + line[1].split('/')[-1])
            imageRight = cv2.imread('data/IMG/' + line[2].split('/')[-1])
            images.extend([imageCenter, imageLeft, imageRight])
            
            # add steering measurements
            correction = 0.02
            steeringCenter = float(line[3])
            steeringLeft = steeringCenter + correction
            steeringRight = steeringCenter - correction
            steeringAngles.extend([steeringCenter, steeringLeft, steeringRight])
           
        print('Finished reading in training data.')
        
    X_train = np.array(images) # shape: (24108, 160, 320, 3)
    y_train = np.array(steeringAngles) # shape: (24108,)
    
    return X_train, y_train
