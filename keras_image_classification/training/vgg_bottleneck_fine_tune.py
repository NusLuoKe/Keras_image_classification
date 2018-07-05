#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/12 16:04
# @File    : vgg_bottleneck_fine_tune.py
# @Author  : NusLuoKe


'''
training image size is 64*64*3
{'cat': 0, 'dog': 1}
'''

import time

from keras import optimizers
from keras.applications import VGG16
from keras.layers import Dense, Flatten
from keras.layers import Dropout
from keras.models import Model
from keras.models import Sequential
from keras.regularizers import l2

from keras_image_classification.visualization_util import *

top_model_weights_path = 'T:/keras_kaggle/models/VGG_bottleneck_feature/bottleneck_fc_model_weights.h5'
bottleneck_features_train_npy = 'T:/keras_kaggle/models/VGG_bottleneck_feature/bottleneck_features_train.npy'
bottleneck_features_validation_npy = 'T:/keras_kaggle/models/VGG_bottleneck_feature/bottleneck_features_validation.npy'

VGG_weights_path = '../keras/examples/vgg16_weights.h5'

# the directory of training set and validation set
train_dir = 'T:/keras_kaggle/data/train'
validation_dir = 'T:/keras_kaggle/data/validation'

# total number of training and validation set.
nb_train_samples = 6000
nb_validation_samples = 1000

# pre-settings
target_size = (64, 64)
batch_size = 16
epochs = 3

# load model
input_shape = (64, 64, 3)
model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)


def save_bottleneck_feature(model):
    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(rescale=1. / 255)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    # this is a generator that will read pictures of training set
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
        directory=train_dir,
        target_size=target_size,
        batch_size=batch_size,
        shuffle=False,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

    bottleneck_features_train = model.predict_generator(train_generator)
    np.save(bottleneck_features_train_npy, bottleneck_features_train)
    print("save bottleneck_features_train")

    # this is a similar generator, for validation data
    validation_generator = val_datagen.flow_from_directory(
        directory=validation_dir,
        target_size=target_size,
        batch_size=batch_size,
        shuffle=False,
        class_mode='binary')

    bottleneck_features_validation = model.predict_generator(validation_generator)
    np.save(bottleneck_features_validation_npy, bottleneck_features_validation)
    print("save bottleneck_features_train")


def train_top_model(epochs):
    train_data = np.load(bottleneck_features_train_npy)
    train_labels = np.array([0] * int(nb_train_samples / 2) + [1] * int(nb_train_samples / 2))

    validation_data = np.load(bottleneck_features_validation_npy)
    validation_labels = np.array([0] * int(nb_validation_samples / 2) + [1] * int(nb_validation_samples / 2))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    h = model.fit(train_data, train_labels,
                  epochs=epochs,
                  batch_size=batch_size,
                  validation_data=(validation_data, validation_labels),
                  verbose=2)
    model.save_weights(top_model_weights_path)
    return h


def VGG_fine_tune(epochs):
    # load VGG16
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    print("Model loaded!")

    # build a classifier model to put on top of the convolutional model
    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001), ))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))

    top_model.load_weights(top_model_weights_path)

    # build a complete model
    model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

    for layer in model.layers[:15]:
        layer.trainable = False

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=5 * 1e-5, momentum=0.9),
                  metrics=['accuracy'])

    # fine tune
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        directory=train_dir,
        target_size=target_size,
        batch_size=batch_size,
        shuffle=False,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

    validation_generator = val_datagen.flow_from_directory(
        directory=validation_dir,
        target_size=target_size,
        batch_size=batch_size,
        shuffle=False,
        class_mode='binary')

    start = time.time()
    h = model.fit_generator(generator=train_generator,
                            epochs=epochs,
                            verbose=2,
                            validation_data=validation_generator,
                            )

    model_path = 'T:/keras_kaggle/models'
    model_name = 'model_VGG_1.h5'
    weights_path = os.path.join(model_path, model_name)
    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    model.save(weights_path)
    end = time.time()
    time_spend = end - start
    print('@ Overall time spend is %.2f seconds.' % time_spend)
    return h


save_bottleneck_feature(model)
print("Bottleneck feature saved!")
time.sleep(5)
print("Begin to train fully connected layers based on bottleneck feature!")
h = train_top_model(epochs=20)
print("Done!")
time.sleep(5)
VGG_fine_tune(50)
