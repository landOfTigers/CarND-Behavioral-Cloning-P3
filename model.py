from readTrainingData import readTrainingData

X_train, y_train = readTrainingData()

from keras.applications.resnet50 import ResNet50
from keras.layers import Input, Lambda, Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K
import tensorflow as tf

rawInput = Input(shape=(160,320,3))
normalizedInput = Lambda(lambda x: x/255.0-0.5)(rawInput)
resizedInput = Lambda(lambda image: K.tf.image.resize_images(image, (197, 197)))(normalizedInput)

resNet = ResNet50(weights='imagenet', include_top=False, input_shape=(197,197,3))
inp = resNet(resizedInput)
out = GlobalAveragePooling2D()(inp)
# new_layer = Dense(512, activation='relu')(out)
predictions = Dense(1)(out)

model = Model(inputs=rawInput, outputs=predictions)

model.compile(loss='mse', optimizer='adam')
model.summary()
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=2)
model.save('model.h5')