'''
Created on Mar 28, 2018

@author: aditya.gangadhare
'''

import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense,Flatten
from keras.layers.convolutional import Convolution2D
import sklearn
from sklearn.model_selection import train_test_split
from tensorflow.contrib.slim.python.slim.nets.resnet_utils import subsample



"""Read csv file"""

lines =[]

with open('D:\\Eclipse-Workspace\\test\\driving_log.csv') as csvFile:
    reader = csv.reader(csvFile)
    next(reader,None)
    for line in reader:
        lines.append(line)

"""Data Preprocessing"""

def normalizeImage(image):
    return image/255.0-0.5

def resizeImage(image):
    return cv2.resize(image, (64, 64))

def cropImage(image):
    return image[60:140,:]

"""Data Augumentation"""

def flipImage(image):
    return cv2.flip(image,1)



def preprocess(lines):
    
    images=[]
    steeringAngles=[]
    correction = 0.2
    for line in lines:
        list_c=line[0].split("/")
        list_l=line[1].split("/")
        list_r=line[2].split("/")
                    
        image_c=cv2.imread("../IMG/"+list_c[-1])
        image_c=cv2.cvtColor(image_c, cv2.COLOR_BGR2RGB)
        image_l=cv2.imread("../IMG/"+list_l[-1])
        image_l=cv2.cvtColor(image_l, cv2.COLOR_BGR2RGB)
        image_r=cv2.imread("../IMG/"+list_r[-1])
        image_r=cv2.cvtColor(image_r, cv2.COLOR_BGR2RGB)     
              
        image_c=normalizeImage(resizeImage(cropImage(image_c)))
        image_l=normalizeImage(resizeImage(cropImage(image_l)))
        image_r=normalizeImage(resizeImage(cropImage(image_r)))
                    
        steeringAngle_c=float(line[3])
        steeringAngle_l=float(line[3])+correction+0.05
        steeringAngle_r=float(line[3])-correction
                    
        flipped_c=flipImage(image_c)
        flipped_l=flipImage(image_l)
        flipped_r=flipImage(image_r)
    
        images.extend([image_c,image_l,image_r,flipped_c,flipped_l,flipped_r])
        steeringAngles.extend([steeringAngle_c,steeringAngle_l,steeringAngle_r,-steeringAngle_c,-steeringAngle_l,-steeringAngle_r])
    
    return images,steeringAngles


"""Splitting the dataset into training and validation sets"""

train_samples, validation_samples = train_test_split(lines, test_size=0.2)


"""Generator"""

def generator(samples, batch_size=32):
    num_samples = len(samples)
    
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_images,batch_steeringAngles= preprocess(samples[offset:offset+batch_size])
            x_train=np.array(batch_images)
            y_train=np.array(batch_steeringAngles)
            yield x_train, y_train

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

"""Model"""

model = Sequential()
model.add(Convolution2D(24, 5, 5,subsample=(2,2),activation="relu",input_shape=(64,64,3)))
model.add(Convolution2D(36, 5, 5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48, 5, 5,subsample=(2,2),activation='relu'))
model.add(Flatten())
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse',optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*6, validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=2)
model.save('model.h5')