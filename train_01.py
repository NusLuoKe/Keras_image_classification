# -*- coding: utf-8 -*-
# @Time    : 2018/1/28 10:51
# @File    : train_01.py
# @Author  : NusLuoKe

import models
import os
from model_util import load_data
import numpy as np

# load data
data_1, label_1 = load_data('T:/Test/cats', 0)
data_2, label_2 = load_data('T:/Test/dogs', 1)

data = np.vstack((data_1, data_2))
label = np.vstack((label_1, label_2))
print(data.shape, label.shape)

num_samples = len(label)
train_val = int(0.2 * num_samples)
x_train = data[:train_val]
y_train = label[:train_val]
x_test = data[train_val:]
y_test = label[train_val:]

batch_size = 32
nb_epoch = 50

input_shape = (128, 128, 3)
model = models.cnn01(input_shape)
# print the model structure
print(model.summary())

# train the model
h = model.fit(x_train, y_train, epochs=nb_epoch, batch_size=batch_size, validation_data=(x_test, y_test),
              shuffle=True, verbose=2)

# save the model to the following directory
model_dir = 'T:/model_01'
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)
model.save(model_dir + '/MODEL_01.h5')
