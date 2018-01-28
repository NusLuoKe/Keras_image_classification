# -*- coding: utf-8 -*-
# @Time    : 2018/1/28 9:40
# @File    : models.py
# @Author  : NusLuoKe

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalMaxPool2D
from keras.layers import Conv2D, MaxPool2D


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
    model.add(Dense(1024))
    model.add(Activation('relu'))

    # output
    model.add(Dense(10))
    model.add(Activation('softmax'))
    adam = keras.optimizers.Adam(lr=1e-4)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def cnn02():
    '''
    lenet
    '''
    model = Sequential()
    model.add(
        Conv2D(6, (5, 5), padding='valid', activation='relu', kernel_initializer='he_normal', input_shape=(32, 32, 3)))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(16, (5, 5), padding='valid', activation='relu', kernel_initializer='he_normal'))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(120, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(84, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(10, activation='softmax', kernel_initializer='he_normal'))
    adam = keras.optimizers.Adam(lr=1e-4)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def cnn03():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(32, 32, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(48, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(48, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(80, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(80, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(80, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(80, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(80, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(GlobalMaxPool2D())
    model.add(Dropout(0.25))

    model.add(Dense(500))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    adam = keras.optimizers.Adam(lr=1e-4)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model