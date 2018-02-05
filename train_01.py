# -*- coding: utf-8 -*-
# @Time    : 2018/1/28 10:51
# @File    : train_01.py
# @Author  : NusLuoKe

import models
import os
from model_util import load_data
import numpy as np

cat_train = 'T:/kaggle/train/cat'
dog_train = 'T:/kaggle/train/dog'

# load data, dog is 1, cat is 0.
data_1, label_1 = load_data(image_dir=cat_train, image_label=0, image_size=128, channels=3)
data_2, label_2 = load_data(image_dir=dog_train, image_label=1, image_size=128, channels=3)

x_train = np.vstack((data_1, data_2))
y_train = np.vstack((label_1, label_2))
print(x_train.shape, y_train.shape)
print(y_train)
# num_samples = len(label_1)
# var_rate = 0.08
# num_var = int(var_rate * num_samples)
#
# cat_test = data_1[:num_var]
# dog_test = data_2[:num_var]
# cat_label_test = label_1[:num_var]
# dog_label_test = label_2[:num_var]
#
# cat_train = data_1[num_var:]
# dog_train = data_2[num_var:]
# cat_label_train = label_1[num_var:]
# dog_label_train = label_2[num_var:]

# x_train = np.vstack((cat_train, dog_train))
# y_train = np.vstack((cat_label_train, dog_label_train))
#
# x_test = np.vstack((cat_test, dog_test))
# y_test = np.vstack((cat_label_test, dog_label_test))
# print(x_test.shape, y_test.shape)
# # print(y_test)

batch_size = 16
nb_epoch = 3

input_shape = (64, 64, 3)
model = models.cnn01(input_shape)

# # print the model structure
# print(model.summary())

# train the model
h = model.fit(x_train, y_train, epochs=nb_epoch, batch_size=batch_size, validation_split=0.08,
              shuffle=True, verbose=2)

# save the model to the following directory
model_dir = 'T:/kaggle/model_01'
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)
model.save(model_dir + '/MODEL_01.h5')
