# -*- coding: utf-8 -*-
# @Time    : 2018/1/28 9:40
# @File    : cnn_models.py
# @Author  : NusLuoKe

from keras.layers import Conv2D, MaxPool2D, Merge
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import SGD, rmsprop
from keras.models import Sequential


# model01
def cnn01(input_shape):
    model = Sequential()
    # conv1
    model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding="same", input_shape=input_shape))
    model.add(MaxPool2D(pool_size=2, strides=2))
    model.add(Activation('relu'))

    # con2
    model.add(Conv2D(filters=128, kernel_size=3, strides=1, padding='same'))
    model.add(MaxPool2D(pool_size=2, strides=2))
    model.add(Activation('relu'))

    # conv3
    model.add(Conv2D(filters=256, kernel_size=3, strides=1, padding='same'))
    model.add(MaxPool2D(pool_size=2, strides=2))
    model.add(Activation('relu'))

    # hidden
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))

    # output
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    # sgd =SGD(lr=0.001)
    # model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def cnn02(input_shape):
    # left
    model_left = Sequential()
    model_left.add(Conv2D(filters=64, kernel_size=3, strides=1, padding="same", input_shape=input_shape))
    model_left.add(MaxPool2D(pool_size=2, strides=2))
    model_left.add(Activation('relu'))
    model_left.add(Flatten())
    model_left.add(Dense(64))
    model_left.add(Activation('relu'))

    # middle
    model_middle = Sequential()
    model_middle.add(Conv2D(filters=128, kernel_size=3, strides=1, padding="same", input_shape=input_shape))
    model_middle.add(MaxPool2D(pool_size=2, strides=2))
    model_middle.add(Activation('relu'))
    model_middle.add(Conv2D(filters=256, kernel_size=3, strides=1, padding="same"))
    model_middle.add(MaxPool2D(pool_size=2, strides=2))
    model_middle.add(Activation('relu'))
    model_middle.add(Flatten())
    model_middle.add(Dense(128))
    model_middle.add(Activation('relu'))

    # right
    model_right = Sequential()
    model_right.add(Conv2D(filters=128, kernel_size=3, strides=1, padding="same", input_shape=input_shape))
    model_right.add(MaxPool2D(pool_size=2, strides=2))
    model_right.add(Activation('relu'))
    model_right.add(Conv2D(filters=256, kernel_size=3, strides=1, padding="same"))
    model_right.add(MaxPool2D(pool_size=2, strides=2))
    model_right.add(Activation('relu'))
    model_right.add(Flatten())
    model_right.add(Dense(128))
    model_right.add(Activation('relu'))

    merged_model = Merge([model_left, model_middle, model_right], mode='concat')
    # merged_model = Merge([model_left, model_middle], mode='concat')

    # merge
    final_model = Sequential()
    final_model.add(merged_model)
    # output
    final_model.add(Dense(1))
    final_model.add(Activation('sigmoid'))
    final_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    return final_model


def cnn03(input_shape):
    # left
    model_left = Sequential()
    model_left.add(Conv2D(filters=64, kernel_size=3, strides=1, padding="same", input_shape=input_shape))
    model_left.add(MaxPool2D(pool_size=2, strides=2))
    model_left.add(Activation('relu'))
    model_left.add(Flatten())
    model_left.add(Dense(64))
    model_left.add(Activation('relu'))

    # right
    model_right = Sequential()
    model_right.add(Conv2D(filters=128, kernel_size=3, strides=1, padding="same", input_shape=input_shape))
    model_right.add(MaxPool2D(pool_size=2, strides=2))
    model_right.add(Activation('relu'))
    model_right.add(Conv2D(filters=256, kernel_size=3, strides=1, padding="same"))
    model_right.add(MaxPool2D(pool_size=2, strides=2))
    model_right.add(Activation('relu'))
    model_right.add(Flatten())
    model_right.add(Dense(128))
    model_right.add(Activation('relu'))

    merged_model = Merge([model_left, model_right], mode='concat')

    # merge
    final_model = Sequential()
    final_model.add(merged_model)
    # output
    final_model.add(Dense(1))
    final_model.add(Activation('sigmoid'))
    final_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    return final_model
