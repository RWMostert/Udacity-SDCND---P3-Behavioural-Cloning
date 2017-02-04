#**Behavioural Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[leftImage]: output_9.png "Left Image"
[centerImage]: output_9_1.png "Center Image"
[rightImage]: output_9_2.png "Right Image"

[augmentedOriginalImage]: output_17_0.png "Augmented Original"
[brightnesVarImage]: output_17_1.png "Brightness Varied Image"
[shiftedImage]: output_17_2.png "Randomly Shifted Image"
[flippedImage]: output_17_3.png "Flipped Image"

[convNetStructure]: output_22_0.png "Convnet Structure"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolutional neural network 
* writeup_report.md summarizing the results

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

My model consists of a convolution neural network with 5 convolutional layers and 4 fully-connected layers.
It starts out with a normalization layer, which normalizes pixel values to be between -1 and 1. 
Then follow 3 convolutional layers with a 5x5 kernel each (of sizes 24, 36 and 48), and 2 convolutional layers with a 3x3 kernel each (of size 64)  (model.py lines 117-129).
Lastly we have 4 fully-connected layers, of sizes 1164, 100, 50 and 10.

The model includes ELU layers to introduce nonlinearity (code line 117-146), and the data is normalized in the model using a Keras lambda layer (code line 115). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 124, 131, 136, 139, 143, and 145). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 30, 31 & 157, 158). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 148).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road using the left and right camera images as training data with an adjusted steering angle (+0.25 for left images and -0.25 for right images). (model.py lines 58-65) 

For details about how I created the training data, see the next section. 

###Solution Development Strategy & Final Architecture

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to toy with a convolutional neural network, trained on the data provided in the project description. 

My first step was to use a convolution neural network model similar to the NVIDIA end-to-end-learning model (https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). This seemed to be a good point of reference as it was well-documented and proven to yield good results.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I augmented the training data set, as set out in NVIDIA's paper, and further explained by Dr Vivek Yadav in his blogpost (https://chatbotslife.com/learning-human-driving-behavior-using-nvidias-neural-network-model-and-image-augmentation-80399360efee#.ypdb28rdr).  I applied brightness variation, randomised shifts and flips in an attempt to get the model to generalise beyond the training set provided (and not just memorise the set).


The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I tweaked the data augmentation procedures, changed the model structure (read NVIDIA's paper multiple times to understand what I could do to improve) and toyed with different driving settings (throttle).

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 113-150) consisted of a convolution neural network with the following layers and layer sizes:
- Convolution Layer: 5x5 kernel, size of 24
- Convolution Layer: 5x5 kernel, size of 36
- Convolution Layer: 5x5 kernel, size of 48
- Convolution Layer: 3x3 kernel, size of 64
- Convolution Layer: 3x3 kernel, size of 64
- Fully-connected Layer: size of 1164
- Fully-connected Layer: size of 100
- Fully-connected Layer: size of 50
- Fully-connected Layer: size of 10

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![png](output_22_0.png)

####3. Creation of the Training Set & Training Process

I used Udacity's dataset provided, and tweaked it using data augmentation techniques. Here is an example image of center lane driving:

![alt text][centerImage]

I also used the left and right images to simulate recovery, adding 0.25 and -0.25 to the steering angles respectively.
![alt text][leftImage]
![alt text][rightImage]

To augment the data sat, I also flipped images and angles thinking that this would help the model generalise (model.py lines 49-52). In addition I performed random shifts on the images (and accompanying steering angles), as well as brightness varyations. (model.py lines 37-47)
<br>
For example:

![alt text][augmentedOriginalImage]
![alt text][flippedImage]
![alt text][shiftedImage]
![alt text][brightnesVarImage]


I finally randomly shuffled the data set, kept 6500 examples for the training set and put the rest of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 30 (of 6500 samples each) as evidenced by the driving performance in the simulator. I used an adam optimizer so that manually training the learning rate wasn't necessary.
