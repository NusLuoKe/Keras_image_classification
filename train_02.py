# -*- coding: utf-8 -*-
# @Time    : 2018/2/4 21:14
# @File    : train_02.py
# @Author  : NusLuoKe

'''
training image size is 64*64*3
train model 01
'''
import os

from keras.preprocessing.image import ImageDataGenerator

import models

# the directory of training set and validation set
train_dir = 'T:/keras_kaggle/data/train'
validation_dir = 'T:/keras_kaggle/data/validation'

# pre-settings
target_size = (64, 64)
batch_size = 32
epochs = 50

# load model
input_shape = (64, 64, 3)
model = models.cnn01(input_shape=input_shape)

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

# this is a generator that will read pictures of training set
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
    directory=train_dir,  # this is the target directory
    target_size=target_size,  # all images will be resized to 150x150
    batch_size=batch_size,
    class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
    directory=validation_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='binary')

model.fit_generator(generator=train_generator,
                    epochs=epochs,
                    verbose=2,
                    validation_data=validation_generator,
                    )

model_path = 'T:/keras_kaggle/models'
weights_path = os.path.join(model_path, 'model_01.h5')
if not os.path.isdir(weights_path):
    os.makedirs(weights_path)

model.save_weights(weights_path)
