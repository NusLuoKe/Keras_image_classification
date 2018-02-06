# -*- coding: utf-8 -*-
# @Time    : 2018/2/6 14:43
# @File    : train_04.py
# @Author  : NusLuoKe

'''
use VGG-16 net to capture features
freeze the first 4 convolution blocks and retrain the last convolution block and our own fully connected layers
'''

import os
import time
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
import numpy as np
from keras import optimizers
from keras.applications.vgg16 import VGG16

# path to the model weights files.
weights_path = '../keras/examples/vgg16_weights.h5'
top_model_weights_path = 'fc_model.h5'

# the directory of training set and validation set
train_dir = 'T:/keras_kaggle/data/train'
validation_dir = 'T:/keras_kaggle/data/validation'

num__train_dogs = len(os.listdir(os.path.join(train_dir, 'dog')))
num_train_sample = num__train_dogs * 2
num__test_dogs = len(os.listdir(os.path.join(validation_dir, 'dog')))
num_test_sample = num__test_dogs * 2

# pre-settings
target_size = (64, 64)
batch_size = 32
epochs = 50
input_shape = (64, 64, 3)


def load_model(input_shape):
    model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    top_model = Sequential()
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))

    # note that it is necessary to start with a fully-trained
    # classifier, including the top classifier,
    # in order to successfully do fine-tuning
    top_model.load_weights(top_model_weights_path)

    # add the model on top of the convolutional base
    model.add(top_model)

    print(model.summary())

    # set the first 25 layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    for layer in model.layers[:25]:
        layer.trainable = False

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])
    return model


# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='binary')

# load model
model = load_model(input_shape=input_shape)

# fine-tune the model
model.fit_generator(
    train_generator,
    verbose=2,
    nb_epoch=epochs,
    validation_data=validation_generator,
    nb_val_samples=num_test_sample)

model_path = 'T:/keras_kaggle/models'
weights_path = os.path.join(model_path, 'model_03.h5')
if not os.path.isdir(weights_path):
    os.makedirs(weights_path)

model.save_weights(weights_path)
