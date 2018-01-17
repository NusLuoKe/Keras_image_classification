# -*- coding: utf-8 -*-
# @Time    : 2018/1/17 15:29
# @File    : image_preprocess.py
# @Author  : NusLuoKe


import os
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


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


def resizeImg(**args):
    '''
    :param args:
    :return:

    example：
    ori_img = "D:/ori/cat_origin.jpg"
    dst_img = "D:/dst/cat_resize.jpg"
    dst_w = 94
    dst_h = 94
    save_q = 100

    resizeImg(ori_img=ori_img,dst_img=dst_img,dst_w=dst_w,dst_h=dst_h,save_q = save_q)
    '''
    args_key = {'ori_img': '', 'dst_img': '', 'dst_w': '', 'dst_h': '', 'save_q': 75}
    arg = {}
    for key in args_key:
        if key in args:
            arg[key] = args[key]

    img = Image.open(arg['ori_img'])
    ori_w, ori_h = img.size
    widthRatio = heightRatio = None
    ratio = 1
    if (ori_w and ori_w > arg['dst_w']) or (ori_h and ori_h > arg['dst_h']):
        if arg['dst_w'] and ori_w > arg['dst_w']:
            widthRatio = float(arg['dst_w']) / ori_w  # 正确获取小数的方式
        if arg['dst_h'] and ori_h > arg['dst_h']:
            heightRatio = float(arg['dst_h']) / ori_h

        if widthRatio and heightRatio:
            if widthRatio < heightRatio:
                ratio = widthRatio
            else:
                ratio = heightRatio

        if widthRatio and not heightRatio:
            ratio = widthRatio
        if heightRatio and not widthRatio:
            ratio = heightRatio

        newWidth = int(ori_w * ratio)
        newHeight = int(ori_h * ratio)
    else:
        newWidth = ori_w
        newHeight = ori_h

    img.resize((newWidth, newHeight), Image.ANTIALIAS).save(arg['dst_img'], quality=arg['save_q'])


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

img_dir = "D:/Rename/augment/cat.jpg"
new_img_dir = "D:/Rename/augment/newgen"
img = load_img(img_dir)
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

image_datagen.fit(x, augment=True)
i = 0
for batch in image_datagen.flow(x, batch_size=5,
                                save_to_dir=new_img_dir, save_prefix='dog', save_format='jpeg'):
    i += 1
    if i > 1:
        break