import os
import csv
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2, activity_l2
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import math
import numpy as np
from PIL import Image         
import cv2                 
import matplotlib.pyplot as plt
from os import getcwd
import csv
# Fix error with TF and Keras
import tensorflow as tf
tf.python.control_flow_ops = tf



samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)

    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                img_center = cv2.imread('./data/IMG/'+batch_sample[0].split('/')[-1])
                img_left = cv2.imread('./data/IMG/'+batch_sample[1].split('/')[-1])
                img_right = cv2.imread('./data/IMG/'+batch_sample[2].split('/')[-1])
                
                steering_center = float(batch_sample[3])
                correction = 0.2
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                images.append(img_center)
                images.append(img_left)
                images.append(img_right)
                angles.append(steering_center)
                angles.append(steering_left)
                angles.append(steering_right)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
            yield (X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(160, 320, 3)))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= len(train_samples)*3, validation_data=validation_generator, nb_val_samples=len(validation_samples)*3, nb_epoch=5)
model.save('model.h5')

