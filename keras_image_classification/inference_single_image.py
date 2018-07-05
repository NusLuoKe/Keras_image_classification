# -*- coding: utf-8 -*-
# @Time    : 2018/2/12 16:11
# @File    : inference_single_image.py
# @Author  : NusLuoKe


import os

from keras.models import load_model

from keras_image_classification.visualization_util import pred_one_img

model_dir = 'T:/keras_kaggle/models/'
model_name = 'model_1.h5'
model_path = os.path.join(model_dir, model_name)

model = load_model(model_path)
print('model loaded')

# parameters
class_01 = 'cat'
class_02 = 'dog'
input_shape = (64, 64)
image_dir = 'T:/keras_kaggle/data/11.jpg'

pred_one_img(model=model, image_dir=image_dir, input_shape=input_shape, class_01=class_01, class_02=class_02)