# 1: define model architecture

from keras.applications.mobilenet import MobileNet
from keras.layers import Input, Lambda, Dense, GlobalAveragePooling2D, Cropping2D
from keras.models import Model
from keras import backend as K
import tensorflow as tf

rawInput = Input(shape=(160,320,3))
croppedInput = Cropping2D(cropping=((70,25),(0,0)))(rawInput)
normalizedInput = Lambda(lambda x: x/255.0-0.5)(croppedInput)
resizedInput = Lambda(lambda image: K.tf.image.resize_images(image, (128, 128)))(normalizedInput)
baseModel = MobileNet(include_top=False, weights='imagenet', input_shape=(128,128,3))(resizedInput)
pooling = GlobalAveragePooling2D()(baseModel)
predictions = Dense(1)(pooling)

model = Model(inputs=rawInput, outputs=predictions)
model.compile(loss='mse', optimizer='adam')
model.summary()


# 2: read training data and train the model using the generator function

from createSamplesFromLog import createSamplesFromLog    
trainSamples, validationSamples = createSamplesFromLog()

batchSize=8
from trainingDataGenerator import trainingDataGenerator
trainGenerator = trainingDataGenerator(trainSamples, batchSize)
validationGenerator = trainingDataGenerator(validationSamples, batchSize)

model.fit_generator(trainGenerator, steps_per_epoch=len(trainSamples)/batchSize, validation_data=validationGenerator, validation_steps=len(validationSamples)/batchSize, epochs=2)

model.save('model.h5')
