# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_files/mobileNetArch.png "MobileNet"
[image2]: ./writeup_files/backwards_center.jpg "Backwards Center"
[image3]: ./writeup_files/recovery1.jpg "Recovery 1"
[image4]: ./writeup_files/recovery2.jpg "Recovery 2"
[image5]: ./writeup_files/recovery3.jpg "Recovery 3"
[image6]: ./writeup_files/flipped.png "Flipped"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of the Keras MobileNet with the last fully-connected layer removed and replaced by a fully connected layer with a single output (model.py lines 13-14). The parameter "pooling" is set to "avg" to retain the average pooling layer of the original network. There is the possibility to set the dropout rate via a parameter, however, this was not necessary, as the model was not overfitting the training data.

The input images are cropped, normalized, and resized using Keras cropping and lambda layers (model.py line 10-13). 

Here is an overview of the MobileNet architecture provided from [this paper](https://arxiv.org/pdf/1704.04861v1.pdf):

![alt text][image1]

As we can see, this network architecture uses convolution layers with appropriate filter sizes, and, as we can read in [the paper](https://arxiv.org/pdf/1704.04861v1.pdf), each convolutional layer contains relu sublayers to introduce nonlinearity into the model.

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting. To create and split those data sets I used the function "createSamplesFromLog" which is called from model.py, line 24. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

Staying very close to the original MobileNet architecture, there was no significant overfitting, so the model did not need to be enhanced with additional dropout layers. A curious attempt to set the MobileNet dropout parameter to 0.001 had negative effects on the model's performance and was therefore dismissed.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 17).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and driving the lap counter-clockwise.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to apply transfer learning on one of the models provided in Keras to experinece the power of the networks pretrained on the ImageNet.

My first step was to test the InceptionV3 and ResNet50 models from Keras. I thought these models might be appropriate because they were the winners of the 2014 and 2015 ImageNet competitions, with ResNet's accuracy exceeding that of a human.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

I found that both, the Inception and ResNet models, created very large model files and especially the ResNet model took extremely long to train. The validation loss was around 0.07 after one epoch of training for both of these models and the models would start overfitting with the second epoch. 

Because these complex models felt like an overkill for the task at hand I decided to test one of the smaller and faster ImageNet models provided in Keras, and, after doing some research, decided to look into the MobilNet model. According to this [video](https://www.youtube.com/watch?v=OO4HD-1wRN8) MobileNets are lightweight deep CNNs that are smaller in size and faster in performance than most other popular models.

I experimented with several fully connected layers at the ouput, however, found that the performance on the final model and data was best with only one fully connected layer, which resulted in a validation loss of around 0.024. There was no signifant overfitting problem, however, out of curiosity I still experimented with setting the dropout rate of the Keras model. As expected, it would not contribute to the overall performance.

I found that three epochs were sufficient for training, as the model would always start slightly overfitting from the third epoch.

The final step was to run the simulator to see how well the car is driving around track one. In the earlier stages of the parameter tuning and data collection process, there were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I tuned the correction factor applied to the left and right camera images to 0.02, provided each image in a flipped version, recorded a counter-clockwise lap to combat the drag to the left side of the lane and recorded some examples of recovery driving. The first set of recovery driving apparently included some bad driving behavior examples, which deteriorated the performances. This led me to record some better recovery examples.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 9-14) consisted of a MobileNet with a single fully connected layer at the output and the preprocesssing layers described above at the input.

Here is a summary of the architecture:

| Layer							| Output Shape			| Param #	| 
|:-----------------------------:|:---------------------:|:---------:| 
| input_1 (InputLayer)			| (None, 160, 320, 3)	| 0			|
| cropping2d_1 (Cropping2D)		| (None, 65, 320, 3)	| 0			|
| lambda_1 (Lambda)				| (None, 65, 320, 3)	| 0			|
| lambda_2 (Lambda)				| (None, 128, 128, 3)	| 0			|
| mobilenet_1.00_128 (Model)	| (None, 1024)			| 3228864	|
| dense_1 (Dense)				| (None, 1)				| 1025		|

#### 3. Creation of the Training Set & Training Process

For center lane driving, I used the sample driving data provided through the workspace. Additionally, to combat the left drag, I recorded one counter-clockwise lap of center lane driving. Here is an example of this:

![alt text][image2]

I then recorded the vehicle recovering from the left and right sides of the road back to center so that the vehicle would learn to steer back to the center when it approaches the lane markings. These images show what a recovery looks like starting from the left side with the car dragging over to the center:

![alt text][image3]
![alt text][image4]
![alt text][image5]

To augment the data sat, I also flipped images and angles thinking that this would lead to better generalizations. For example, here is an image and its flipped version:

![alt text][image6]

I also used the left and right camera images with a correction factor of 0.02 which greatly improved the car's ability to stay on the track.

After the collection process, I had 112,962 number of data points. I then preprocessed this data by cropping, normalizing and resizing the images to remove irrelevant information and speed up training.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was three, as evidenced by the difference in training and validation loss, where the validation loss started slightly exceeding the training loss starting from the third epoch. Here is an example output from training with four epochs, where this becomes obvious:

Epoch 1/4  
1883/1882 [==============================] - 502s - loss: 0.0828 - val_loss: 0.0244  
Epoch 2/4  
1883/1882 [==============================] - 499s - loss: 0.0257 - val_loss: 0.0238  
Epoch 3/4  
1883/1882 [==============================] - 499s - loss: 0.0247 - val_loss: 0.0287  
Epoch 4/4  
1883/1882 [==============================] - 499s - loss: 0.0240 - val_loss: 0.0252

I used an adam optimizer so that manually training the learning rate wasn't necessary. For training I used a generator function called "trainingDataGenerator" with a batchSize parameter of eight, which resulted in an actual batch size of 24, because for each data point there were three images and each of them was additionally flipped, resulting in six training examples per data point.

I hosted my training data on Google Drive and programmed a bash script ("fetch-data.sh") to pull the data from there. Unfortunately, the URLs to download the data from Google Drive are only valid for a couple of hours and have to be re-generated and replaced in the script every time they become invalid.
