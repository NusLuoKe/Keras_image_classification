# -*- coding: utf-8 -*-
# @Time    : 2018/1/28 9:40
# @File    : model_util.py
# @Author  : NusLuoKe


import os

import numpy as np
from PIL import Image


def load_data(image_dir, img_label, img_size=128, channels=3):
    images = os.listdir(image_dir)
    num_samples = len(images)
    data = np.empty((num_samples, img_size, img_size, channels), dtype='float32')
    labels = np.empty((num_samples, 1), dtype='uint8')

    for i in range(num_samples):
        img_dir = os.path.join(image_dir, images[i])
        print(img_dir)
        img = Image.open(img_dir)

        arr = np.asarray(img, dtype="float32")
        arr.resize((img_size, img_size, channels))
        data[i, :, :, :] = arr

        labels[i] = img_label
        labels = np.reshape(labels, (num_samples, 1))
        return data, labels


data_1, label_1 = load_data('T:/Test/cats', 0)
print(label_1)
# data_2, label_2 = load_data('T:/Test/dogs', 1)

# print(data_1.shape)
# print(label_1.shape)
# print(label_1)
# print(data_2.shape)
# print(label_2.shape)
# print('#####################')
# print('#####################')
#
# data = np.vstack((data_1, data_2))
# label = np.vstack((label_1, label_2))
# print(data.shape, label.shape)
#
# print('#####################')
# print(label)
