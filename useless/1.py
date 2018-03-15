#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/15 18:17
# @File    : 1.py
# @Author  : NusLuoKe

from math import ceil
import time
from done import my_models
from done.result_visualize_util import *

# the directory of training set and validation set
train_dir = 'T:/keras_kaggle/data/train'
validation_dir = 'T:/keras_kaggle/data/validation'

num_train_samples = len(os.listdir(os.path.join(train_dir, os.listdir(train_dir)[0])))
num_validation_samples = len(os.listdir(os.path.join(validation_dir, os.listdir(validation_dir)[0])))

# pre-settings
target_size = (64, 64)
batch_size = 16
epochs = 3
# load model
input_shape = (64, 64, 3)
model = my_models.cnn03(input_shape=input_shape)

input_imgen = ImageDataGenerator(rescale=1. / 255)

test_imgen = ImageDataGenerator(rescale=1. / 255)


def generate_generator_multiple(generator, dir1, dir2, batch_size, target_size):
    genX1 = generator.flow_from_directory(dir1,
                                          target_size=target_size,
                                          class_mode='binary',
                                          batch_size=batch_size,
                                          shuffle=False,
                                          seed=7)

    genX2 = generator.flow_from_directory(dir2,
                                          target_size=target_size,
                                          class_mode='binary',
                                          batch_size=batch_size,
                                          shuffle=False,
                                          seed=7)
    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        yield [X1i[0], X2i[0]], X2i[1]  # Yield both images and their mutual label


inputgenerator = generate_generator_multiple(generator=input_imgen,
                                             dir1=train_dir,
                                             dir2=train_dir,
                                             batch_size=batch_size,
                                             target_size=target_size)

testgenerator = generate_generator_multiple(test_imgen,
                                            dir1=validation_dir,
                                            dir2=validation_dir,
                                            batch_size=batch_size,
                                            target_size=target_size)

history = model.fit_generator(inputgenerator,
                              steps_per_epoch=int(ceil(num_train_samples / batch_size)),
                              epochs=epochs,
                              validation_data=testgenerator,
                              validation_steps=int(ceil(num_validation_samples / batch_size)),
                              verbose=2
                              )

