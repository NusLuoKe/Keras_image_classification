# -*- coding: utf-8 -*-
# @Time    : 2018/1/17 15:29
# @File    : image_preprocess.py
# @Author  : NusLuoKe


import os
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


def resize_img(ori_img, dst_img, dst_w, dst_h, save_q=75):
    '''
    :param ori_img: path of the original image
    :param dst_img: path of the new image to be stored
    :param dst_w: desitination width of the new image
    :param dst_h: destination height of the new image
    :param save_q: 0 - 100, image quality
    :return:
    '''
    img = Image.open(ori_img)
    ori_w, ori_h = img.size
    width_ratio = height_ratio = None
    ratio = 1
    if (ori_w and ori_w > dst_w) or (ori_h and ori_h > dst_h):
        if dst_w and ori_w > dst_w:
            width_ratio = float(dst_w) / ori_w  # 正确获取小数的方式
        if dst_h and ori_h > dst_h:
            height_ratio = float(dst_h) / ori_h

        if width_ratio and height_ratio:
            if width_ratio < height_ratio:
                ratio = width_ratio
            else:
                ratio = height_ratio

        if width_ratio and not height_ratio:
            ratio = width_ratio
        if height_ratio and not width_ratio:
            ratio = height_ratio

        new_width = int(ori_w * ratio)
        new_height = int(ori_h * ratio)
    else:
        new_width = ori_w
        new_height = ori_h

    img.resize((new_width, new_height), Image.ANTIALIAS).save(dst_img, quality=save_q)


def clip_resize_img(ori_img, dst_img, dst_w, dst_h, save_q=75):
    '''
    :param ori_img: path of the original image
    :param dst_img: path of the new image to be stored
    :param dst_w: desitination width of the new image
    :param dst_h: destination height of the new image
    :param save_q: 0 - 100, image quality
    :return:
    '''
    im = Image.open(ori_img)
    ori_w, ori_h = im.size

    dst_scale = float(dst_h) / dst_w  # destination aspect ratio
    ori_scale = float(ori_h) / ori_w  # original aspect ratio

    # too high or too wide: use too high as an example:
    # keep the width unchanged, calculate the destination height by apply the equation: height = width * dst_scale
    # then calculate how high of the image needs to be cut, that is： ori_h - height
    # at last, cut y = （ori_h - height） / 2 each from the top and bottom
    if ori_scale >= dst_scale:
        width = ori_w
        height = int(width * dst_scale)

        x = 0
        y = (ori_h - height) / 2

    else:
        height = ori_h
        width = int(height * dst_scale)

        x = (ori_w - width) / 2
        y = 0

    # Crop picture, make sure the new image has correct specified aspect ratio.
    # initial point is (x, y), end point is (width + x, height + y), keep images in the box area
    box = (x, y, width + x, height + y)
    new_im = im.crop(box)
    im.close()

    # Compress the picture to the specified size
    ratio = float(dst_w) / width
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    new_im.resize((new_width, new_height), Image.ANTIALIAS).save(dst_img, quality=save_q)


def generate_image(img_dir, save_dir, prefix, gen_num=1, save_format='jpeg'):
    '''
    :param img_dir: the directory of images need to be augmented
    :param save_dir: the directory of augmented images to be stored
    :param prefix: prefix of the name of augmented images
    :param save_format: format of augmented images. 'jepg' or 'png'
    :return:
    '''
    image_datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect')

    if os.path.isdir(save_dir):
        pass
    else:
        os.makedirs(save_dir)

    for file in os.listdir(img_dir):
        if os.path.isdir(file):
            continue
        else:

            image_dir = os.path.join(img_dir, file)
            img = load_img(image_dir)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)

            image_datagen.fit(x, augment=True)
            i = 0
            for _ in image_datagen.flow(x, save_to_dir=save_dir, save_prefix=prefix, save_format=save_format):
                i += 1
                if i >= gen_num:
                    break


def rename_files(file_dir, new_prefix):
    '''
    :param file_dir: The directory of the files which need to be renamed
    :param new_prefix: the prefix of new files
    :return:

    example:
    rename_files('D:/cat', 'cat'): files in 'D:/cat' will be named as 'cat_0', 'cat_1', 'cat_2' ...
    '''
    num = 0
    for file in os.listdir(file_dir):
        oldDir = os.path.join(file_dir, file)
        if os.path.isdir(oldDir):
            continue;
        filetype = os.path.splitext(file)[1]
        n = str(num)
        newName = new_prefix + '_' + n
        newDir = os.path.join(file_dir, newName + filetype)
        os.rename(oldDir, newDir)
        num += 1
