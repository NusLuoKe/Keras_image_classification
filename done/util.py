# -*- coding: utf-8 -*-
# @Time    : 2018/2/7 13:20
# @File    : util.py
# @Author  : NusLuoKe

import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical

from done.image_preprocess_util import *


def visualize_prediciton(model, validation_generator, image_size):
    '''
    :param model: training model
    :param x_test: test set images
    :param y_test: test set labels
    :return: plot a figure of 10 categories different images with their truly category and the model prediction
    '''
    num_show_img = 10

    num_batch = len(validation_generator)
    # num_image_per_batch = len(validation_generator[0][1])
    rand_batch = np.random.randint(num_batch)

    x_test = validation_generator[rand_batch][0]

    y_test = validation_generator[rand_batch][1]
    y_test = to_categorical(y_test, num_classes=2)

    class_name = {
        0: 'cat',
        1: 'dog',
    }

    # rand_id = np.random.choice(range(num_image_per_batch), size=num_show_img)
    rand_id = np.array(range(10))
    y_true = [y_test[i] for i in rand_id]
    y_true = np.argmax(y_true, axis=1)
    y_true = [class_name[name] for name in y_true]

    x_pred = np.array([x_test[i] for i in rand_id])
    y_pred = model.predict(x_pred)

    y_pred_ = []
    for pred in y_pred:
        if pred[0] < 0.5:
            pred[0] = 0
        else:
            pred[0] = 1
        y_pred_.append(pred[0])
    y_pred = y_pred_

    y_pred = [class_name[name] for name in y_pred]

    plt.figure(figsize=(15, 7))
    for i in range(num_show_img):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_pred[i].reshape(image_size))
        if y_true[i] == y_pred[i]:
            plt.title('True: %s \n Pred: %s' % (y_true[i], y_pred[i]), size=15)
        else:
            plt.title('True: %s \n Pred: %s' % (y_true[i], y_pred[i]), size=15, color='red')
    plt.show()


def plot_acc_loss(h, nb_epoch):
    '''
    :param h: history, it is the return value of "fit()", h = model.fit()
    :param nb_epoch: number of epochs
    :return: plot a figure of accuracy and loss of very epoch
    '''
    acc, loss, val_acc, val_loss = h.history['acc'], h.history['loss'], h.history['val_acc'], h.history['val_loss']
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.plot(range(nb_epoch), acc, label='Train')
    plt.plot(range(nb_epoch), val_acc, label='Test')
    plt.title('Accuracy over ' + str(nb_epoch) + ' Epochs', size=15)
    plt.legend()
    plt.grid(True)
    plt.subplot(122)
    plt.plot(range(nb_epoch), loss, label='Train')
    plt.plot(range(nb_epoch), val_loss, label='Test')
    plt.title('Loss over ' + str(nb_epoch) + ' Epochs', size=15)
    plt.legend()
    plt.grid(True)
    plt.show()


def test_batch_gen(test_dir, test_class01_dir, target_size):
    test_batch_len = len(os.listdir(test_class01_dir))
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        directory=test_dir,
        target_size=target_size,
        batch_size=test_batch_len,
        class_mode='binary')

    x_test = test_generator[0][0]
    y_test = test_generator[0][1]

    return x_test, y_test

def pred_one_img(model, image_dir, input_shape=(64, 64), class_01='class_01', class_02='class_02'):
    dst_w = int(input_shape[0])
    dst_h = int(input_shape[1])
    temp_path_ = os.path.split(image_dir)
    temp_path = temp_path_[0]
    temp_name01 = 'pred_one_image_.jpg'
    temp_name02 = 'pred_one_image.jpg'
    temp_dir01 = os.path.join(temp_path, temp_name01)
    temp_dir02 = os.path.join(temp_path, temp_name02)

    resize_img(image_dir, temp_dir01, dst_w, dst_h, save_q=100)
    clip_resize_img(temp_dir01, temp_dir02, dst_w, dst_h, save_q=100)

    img = Image.open(temp_dir02)
    img_arr = np.asarray(img, dtype="float32")
    x_ = np.expand_dims(img_arr, axis=0)
    x = x_ / 255

    y = model.predict(x)
    y_pred = y[0][0]

    ori_img = Image.open(image_dir)
    plt.figure()
    plt.imshow(ori_img)
    if y_pred > 0.5:
        print("This is a image of", class_01, 'with a probability of: %.2f' % (y_pred * 100) + '%')
        plt.title('%s \n Prob: %.2f%%' % (class_01, y_pred * 100))
    else:
        y_pred = 1 - y_pred
        print("This is a image of", class_02, 'with a probability of: %.2f' % (y_pred * 100) + '%')
        plt.title('%s \n Prob: %.2f%%' % (class_02, y_pred * 100))
    plt.show()

    os.remove(temp_dir01)
    os.remove(temp_dir02)

