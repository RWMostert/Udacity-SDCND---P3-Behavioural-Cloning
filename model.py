import cv2
import json
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from numpy.random import uniform, randint
from pathlib import Path
from scipy import signal

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Lambda
from keras.layers import ELU, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam

tf.python.control_flow_ops = tf

"""
Import the CSV file and keep only the data with a throttle of greater than .25 applied. 
This ensures greater consistency and a smaller propensity of the model to tend towards zero values.
Next We shuffle the data and split it into a testing set and a training set.
"""

df = pd.read_csv('./data/driving_log.csv', header = 0)
df = df[df['throttle']>.25].append(df.iloc[50:110])
shuffled_df = df.sample(len(df)).reset_index()
df_train = shuffled_df[:6500]
df_valid = shuffled_df[6500:]

"""
As described in NVIDIA's paper on end-to-end-learning (https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/), we augment the training data provided by varying the brightness, randomly shifting the image and randomly flipping the image. <br>
"""

def set_rdm_brightness(image):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv_img[:,:,2] = hsv_img[:,:,2]*uniform(low=0.25, high=1.25)
    return cv2.cvtColor(hsv_img,cv2.COLOR_HSV2RGB)

def rdm_shift_image(image, steering):
    Δx = uniform(low = -1, high = 1)
    Δy = uniform(low = -5, high = 5)
    translation = np.float32([ [1, 0, Δx*75], [0, 1, Δy] ])
    steering += Δx/5
    return cv2.warpAffine(image, translation, image.shape[:2][::-1]), steering

def rdm_flip_image(image, steering_angle):
    if uniform() < 0.5:
        return cv2.flip(image,1), -steering_angle
    return image, steering_angle

def image_and_steering_from_file(file, center_only = False):
    image = cv2.imread('./data/' + file['center'])
    steering = file['steering']
    
    if not center_only:
        rv = uniform()
        if rv < 1/3:
            image = cv2.imread('./data/' + file['left'].lstrip())
            steering += .25
        elif rv < 2/3:
            image = cv2.imread('./data/' + file['right'].lstrip())
            steering += -.25
        
    return cv2.cvtColor(image,cv2.COLOR_BGR2RGB), steering

"""
As mentioned in the Nvidia paper, we crop the image to exclude the car and the sky.
I then resize the image to a square 66x66 image (at 66x200 - the size used in the Nvidia paper - my model size easily exceeded 200mb, which would rapidly deplete my data cap allowances when downloading the model from AWS).
"""

def crop_and_resize(image):
    shape = image.shape
    image = image[40:shape[0]-25,:]
    image = cv2.resize(image,(66,66), interpolation=cv2.INTER_AREA) 
    return np.array(image)

"""
Next we set up a batch generator so that all the images need not be kept in memory at training time. One for training and one for validation.
"""
def batch_generator_train(data, batch_size = 32):
    while True:
        images = np.zeros((batch_size, 66, 66, 3))
        steering_values = np.zeros(batch_size)
        
        for i in range(batch_size):
            file = data.iloc[randint(len(data))]
            image, steering = rdm_flip_image(*rdm_shift_image(*image_and_steering_from_file(file)))
            image = crop_and_resize(set_rdm_brightness(image))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            
            images[i] = image
            steering_values[i] = steering
            
        yield images, steering_values
def batch_generator_validation(data):
    while True:
        for i in range(len(data)):
            file = data.iloc[i]
            image, steering = image_and_steering_from_file(file, center_only = True)
            image = crop_and_resize(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            
            yield image.reshape(1, *image.shape), np.array([[steering]])

"""
Next is the building of the actual model, using Keras (with a tensorflow backend).
The model is built after the model described in NVIDIA's paper. <br>
It consists of a typical convolutional neural network, with 5 convolutional layers and 4 fully-connected layers.  It starts out with a normalization layer, which normalizes pixel values to be between -1 and 1.  Then follow 3 convolutional layers with a 5x5 kernel each, and 2 convolutional layers with a 3x3 kernel each. We use an ELU to provide some nonlinearity between the layers; and - in order to reduce over-fitting, we use dropout throughout the network.
"""
def convnet():
	model = Sequential()
	model.add(Lambda(lambda x: x/255.-0.5, input_shape=(66, 66, 3)))

	model.add(Convolution2D(24, 5,5, subsample=(2,2)))
	model.add(ELU())
	model.add(Convolution2D(36, 5,5, subsample=(2,2)))
	model.add(ELU())
	model.add(Convolution2D(48, 5,5))
	model.add(ELU())

	model.add(Dropout(0.5))

	model.add(Convolution2D(64, 3,3))
	model.add(ELU())
	model.add(Convolution2D(64, 3,3))
	model.add(ELU())
		  
	model.add(Dropout(0.5))
		  
	model.add(Flatten())
	model.add(Dense(1164))
	model.add(ELU())
	model.add(Dropout(0.5))
	model.add(Dense(100))
	model.add(ELU())
	model.add(Dropout(0.5))
	model.add(Dense(50))
	model.add(ELU())
	model.add(Dropout(0.5))
	model.add(Dense(10))
	model.add(ELU())
	model.add(Dropout(0.5))
	model.add(Dense(1))

	model.compile(optimizer=Adam(), loss='mse')

	return model

"""
Using the Adam optimizer, we can now fit the convnet to our data. 
Using my generators defined earlier, I generate training batches of length 128 each, and fit the model with 30 epochs of sample length equal to the training set (6000). 
I validate the training using the validation set generator I defined earlier.
"""
train_gen = batch_generator_train(df_train,128)
valid_gen = batch_generator_validation(df_valid)

model = convnet()
history = model.fit_generator(train_gen, samples_per_epoch=len(df_train), nb_epoch=30, validation_data = valid_gen, nb_val_samples = len(df_valid))
model.save('Model.h5')
