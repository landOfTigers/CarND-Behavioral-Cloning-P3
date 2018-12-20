def trainingDataGenerator(samples, batchSize):
    import cv2
    import numpy as np
    from flipTrainingData import flipTrainingData
    numSamples = len(samples)
    while 1:
        for offset in range(0, numSamples, batchSize):
            batchSamples = samples[offset:offset+batchSize]

            images = []
            steeringAngles = []
            for line in batchSamples:              
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

            # flip images
            X_train, y_train = flipTrainingData(np.array(images), np.array(steeringAngles))

            yield X_train, y_train