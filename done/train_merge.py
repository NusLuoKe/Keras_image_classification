#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/13 22:17
# @File    : train_merge.py
# @Author  : NusLuoKe

'''
training image size is 64*64*3
{'cat': 0, 'dog': 1}
train model 01
'''
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
epochs = 8
# load model
input_shape = (64, 64, 3)
model = my_models.cnn03(input_shape=input_shape)

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)


# this is the augmentation configuration we will use for testing:
# only rescaling
val_datagen = ImageDataGenerator(rescale=1. / 255)


def my_two_inputgenerator(generator, data_dir_01, data_dir_02, batch_size, target_size):
    # this is a generator that will read pictures of training set
    # batches of augmented image data
    generator_01 = generator.flow_from_directory(
        directory=data_dir_01,
        target_size=target_size,
        batch_size=batch_size,
        shuffle=True,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

    generator_02 = generator.flow_from_directory(
        directory=data_dir_02,
        target_size=target_size,
        batch_size=batch_size,
        shuffle=True,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

    while True:
        gen01 = generator_01.next()
        gen02 = generator_02.next()
        yield ([gen01[0], gen02[0]], gen01[1])


final_train_generator = my_two_inputgenerator(generator=train_datagen, data_dir_01=train_dir, data_dir_02=train_dir,
                                              batch_size=batch_size, target_size=target_size)
print(final_train_generator[1])
print("@@@@@@@@@@@@@@@@@@@@@@@")
final_validation_generator = my_two_inputgenerator(generator=val_datagen, data_dir_01=validation_dir,
                                                   data_dir_02=validation_dir,
                                                   batch_size=batch_size, target_size=target_size)
print(".....................")

start = time.time()
h = model.fit_generator(generator=final_train_generator,
                        steps_per_epoch=int(ceil(num_train_samples / batch_size)),
                        epochs=epochs,
                        verbose=2,
                        validation_data=final_validation_generator,
                        validation_steps=int(ceil(num_validation_samples / batch_size))
                        )

# model_path = 'T:/keras_kaggle/models'
# model_name = 'model_merge_1.h5'
# weights_path = os.path.join(model_path, model_name)
# if not os.path.isdir(model_path):
#     os.makedirs(model_path)
#
# model.save(weights_path)
# end = time.time()
# time_spend = end - start
# print('@ Overall time spend is %.2f seconds.' % time_spend)
