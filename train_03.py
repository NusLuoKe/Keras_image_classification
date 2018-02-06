# -*- coding: utf-8 -*-
# @Time    : 2018/2/5 22:20
# @File    : train_03.py
# @Author  : NusLuoKe

'''
Use the convolution blocks of VGG-16 to capture image bottleneck features
for the training set and test set and save as numpy array

And load the features to do further training works of fully connect layers
'''
import os
import time
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from keras.applications.vgg16 import VGG16

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


def save_bottleneck_features(input_shape):
    # load VGG16 which did not contains the top 3 fully connected layers to capture features
    model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    print('Model loaded！')

    datagen = ImageDataGenerator(rescale=1. / 255)

    # training image generator
    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    print('Training set generator done!')

    # get bottleneck feature for training set
    bottleneck_features_train = model.predict_generator(train_generator, steps=num_train_sample)
    print('Training set predict generator done!.')

    np.save(open('bottleneck_features_train.npy', 'wr'), bottleneck_features_train)

    # test image generator
    test_generator = datagen.flow_from_directory(
        validation_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    print('Test set generator done!')

    # get bottleneck feature for test set
    bottleneck_features_validation = model.predict_generator(test_generator, steps=num_test_sample)
    print('Test set predict generator done!.')

    np.save(open('bottleneck_features_validation.npy', 'wr'), bottleneck_features_validation)


from keras.layers import Flatten, Dropout, Dense


def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy'))
    train_labels = np.array([0] * (num_train_sample / 2) + [1] * (num_train_sample / 2))

    validation_data = np.load(open('bottleneck_features_validation.npy'))
    validation_labels = np.array([0] * (num_test_sample / 2) + [1] * (num_test_sample / 2))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs, batch_size=32,
              validation_data=(validation_data, validation_labels))

    model_path = 'T:/keras_kaggle/models'
    weights_path = os.path.join(model_path, 'model_02.h5')
    if not os.path.isdir(weights_path):
        os.makedirs(weights_path)

    model.save_weights(weights_path)

start = time.time()
save_bottleneck_features(input_shape=input_shape)
end = time.time()
print('@ overall time spend is %.2f seconds' % (end - start))
