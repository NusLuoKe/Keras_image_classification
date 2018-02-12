# -*- coding: utf-8 -*-
# @Time    : 2018/2/4 21:14
# @File    : train_basic.py
# @Author  : NusLuoKe

'''
training image size is 64*64*3
{'cat': 0, 'dog': 1}
train model 01
'''
import time

from done import models
from done.util import *

# the directory of training set and validation set
train_dir = 'T:/keras_kaggle/data/train'
validation_dir = 'T:/keras_kaggle/data/validation'

# pre-settings
target_size = (64, 64)
batch_size = 16
epochs = 100
# load model
input_shape = (64, 64, 3)
model = models.cnn01(input_shape=input_shape)

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
val_datagen = ImageDataGenerator(rescale=1. / 255)

# this is a generator that will read pictures of training set
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = val_datagen.flow_from_directory(
    directory=validation_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='binary')

start = time.time()
h = model.fit_generator(generator=train_generator,
                        epochs=epochs,
                        verbose=2,
                        validation_data=validation_generator,
                        )

model_path = 'T:/keras_kaggle/models'
model_name = 'model_1.h5'
weights_path = os.path.join(model_path, model_name)
if not os.path.isdir(model_path):
    os.makedirs(model_path)

model.save(weights_path)
end = time.time()
time_spend = end - start
print('@ Overall time spend is %.2f seconds.' % time_spend)

# plot figures of accuracy and loss of every epoch and a visible test result
plot_acc_loss(h, epochs)
visualize_prediciton(model, validation_generator, image_size=(64, 64, 3))

# # print loss and accuracy on the whole training set and test set
test_dir = 'T:/keras_kaggle/data/test'
test_class01_dir = 'T:/keras_kaggle/data/test/dog'
x_test, y_test = test_batch_gen(test_dir, test_class01_dir, target_size)

print("Test batch generated!Please wait for the final test result:")
loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
print("Testing Accuracy = %.2f %%    loss = %f" % (accuracy * 100, loss))
