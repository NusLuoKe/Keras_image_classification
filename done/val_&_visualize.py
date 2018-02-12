# -*- coding: utf-8 -*-
# @Time    : 2018/2/8 20:38
# @File    : val_&_visualize.py
# @Author  : NusLuoKe

from keras import models

from done.util import *

# Test set directory
test_dir = 'T:/keras_kaggle/data/test'

target_size = (64, 64)
image_size = (64, 64, 3)
batch_size = 64


test_datagen = ImageDataGenerator(rescale=1. / 255)

# this is a similar generator, for validation data
test_generator = test_datagen.flow_from_directory(
    directory=test_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='binary')
print('done')

model_dir = 'T:/keras_kaggle/models/'
model_name = 'model_1.h5'
model_path = os.path.join(model_dir, model_name)

model = models.load_model(model_path)
print('model loaded')

visualize_prediciton(model, test_generator, image_size)

# # print loss and accuracy on the whole training set and test set
test_dir = 'T:/keras_kaggle/data/test'
test_class01_dir = 'T:/keras_kaggle/data/test/dog'
x_test, y_test = test_batch_gen(test_dir, test_class01_dir, target_size)

print("Test batch generated!Please wait for the final test result:")
loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
print("Testing Accuracy = %.2f %%    loss = %f" % (accuracy * 100, loss))
