#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/12 16:04
# @File    : train_VGG_bottleneck.py
# @Author  : NusLuoKe


'''
training image size is 64*64*3
{'cat': 0, 'dog': 1}
'''

from keras.applications import VGG16
from keras.layers import Dense, Flatten
from keras.layers import Dropout
from keras.models import Sequential

from done.result_visualize_util import *

top_model_weights_path = 'T:/keras_kaggle/models/VGG_bottleneck_feature/bottleneck_fc_model_weights.h5'
bottleneck_features_train_npy = 'T:/keras_kaggle/models/VGG_bottleneck_feature/bottleneck_features_train.npy'
bottleneck_features_validation_npy = 'T:/keras_kaggle/models/VGG_bottleneck_feature/bottleneck_features_validation.npy'

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


def train_top_model():
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


# save_bottleneck_feature(model)
# print("Bottleneck feature saved!")
# time.sleep(5)
# print("Begin to train fully connected layers based on bottleneck feature!")
h = train_top_model()
print("Done!")

# plot figures of accuracy and loss of every epoch and a visible test result
plot_acc_loss(h, epochs)
