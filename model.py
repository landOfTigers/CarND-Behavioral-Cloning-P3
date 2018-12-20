from readTrainingData import readTrainingData
from flipTrainingData import flipTrainingData
import numpy as np

# X_train, y_train = readTrainingData()
# np.save('/opt/carnd_p3/tmp/xtrain.npy', X_train)
# np.save('/opt/carnd_p3/tmp/ytrain.npy', y_train)
X_train = np.load('/opt/carnd_p3/tmp/xtrain.npy')
y_train = np.load('/opt/carnd_p3/tmp/ytrain.npy')

X_train, y_train = flipTrainingData(X_train, y_train)

from keras.applications.resnet50 import ResNet50
from keras.layers import Input, Lambda, Dense, GlobalAveragePooling2D, Cropping2D
from keras.models import Model
from keras import backend as K
import tensorflow as tf

rawInput = Input(shape=(160,320,3))
croppedInput = Cropping2D(cropping=((70,25),(0,0)))(rawInput)
normalizedInput = Lambda(lambda x: x/255.0-0.5)(croppedInput)
resizedInput = Lambda(lambda image: K.tf.image.resize_images(image, (197, 197)))(normalizedInput)
baseModel = ResNet50(weights='imagenet', include_top=False, input_shape=(197,197,3))(resizedInput)
pooling = GlobalAveragePooling2D()(baseModel)
# new_layer = Dense(512)(out)#, activation='relu')(pooling)
predictions = Dense(1)(pooling)

model = Model(inputs=rawInput, outputs=predictions)

model.compile(loss='mse', optimizer='adam')
model.summary()
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=1)
model.save('model.h5')