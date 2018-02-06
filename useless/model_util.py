# -*- coding: utf-8 -*-
# @Time    : 2018/1/28 9:40
# @File    : model_util.py
# @Author  : NusLuoKe


import os

import numpy as np
from PIL import Image


def load_data(image_dir, image_label, image_size=128, channels=3):
    images = os.listdir(image_dir)
    num_samples = len(images)
    data = np.empty((num_samples, image_size, image_size, channels), dtype='float32')
    labels = np.empty((num_samples, 1), dtype='uint8')

    for i in range(num_samples):
        img_dir = os.path.join(image_dir, images[i])
        img = Image.open(img_dir)

        arr = np.asarray(img, dtype="float32")
        arr.resize((image_size, image_size, channels))
        data[i, :, :, :] = arr

        labels[i] = image_label
        labels = np.reshape(labels, (num_samples, 1))

    return data, labels

