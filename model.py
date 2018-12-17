# The script used to create and train the model.
import csv
import cv2
import numpy as np

images = []
measurements = []
with open('data/driving_log.csv') as csvfile:
    reader = iter(csv.reader(csvfile))
    next(reader) # skip first line
    for line in reader:
        sourcePath = line[0]
        fileName = sourcePath.split('/')[-1]
        currentPath = 'data/IMG/' + fileName
        images.append(cv2.imread(currentPath))
        measurements.append(float(line[3]))

X_train = np.array(images) # shape: (8036, 160, 320, 3)
y_train = np.array(measurements) # shape: (8036,)


from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)
model.save('model.h5')
