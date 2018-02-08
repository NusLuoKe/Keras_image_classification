# -*- coding: utf-8 -*-
# @Time    : 2018/2/4 13:12
# @File    : kaggle_input_images.py
# @Author  : NusLuoKe

from done import image_preprocess_util
import os
import time

dog_ori_dir = 'T:/kaggle/dog_train'
cat_ori_dir = 'T:/kaggle/cat_train'

dog_dst_dir = 'T:/kaggle/train/dog'
cat_dst_dir = 'T:/kaggle/train/cat'

dog_temp_dir = 'T:/kaggle/train/temp_dog'
cat_temp_dir = 'T:/kaggle/train/temp_cat'

clip_resize = 80
resize = 64

dirs = [dog_dst_dir, cat_dst_dir, dog_temp_dir, cat_temp_dir]
for dir in dirs:
    if not os.path.isdir(dir):
        os.makedirs(dir)

start = time.time()

print('Start to clip_resize_images, please wait....')
for image in os.listdir(dog_ori_dir):
    ori_img = os.path.join(dog_ori_dir, image)
    dst_img = os.path.join(dog_temp_dir, image)
    image_preprocess_util.clip_resize_img(ori_img=ori_img, dst_img=dst_img, dst_w=clip_resize, dst_h=clip_resize,
                                          save_q=100)

for image in os.listdir(cat_ori_dir):
    ori_img = os.path.join(cat_ori_dir, image)
    dst_img = os.path.join(cat_temp_dir, image)
    image_preprocess_util.clip_resize_img(ori_img=ori_img, dst_img=dst_img, dst_w=clip_resize, dst_h=clip_resize,
                                          save_q=100)


print('Finished clip_resize_images. Start to resize images, please wait....')
for clip_image in os.listdir(dog_temp_dir):
    clip_img = os.path.join(dog_temp_dir, clip_image)
    resize_img = os.path.join(dog_dst_dir, clip_image)
    image_preprocess_util.resize_img(ori_img=clip_img, dst_img=resize_img, dst_w=resize, dst_h=resize)


for clip_image in os.listdir(cat_temp_dir):
    clip_img = os.path.join(cat_temp_dir, clip_image)
    resize_img = os.path.join(cat_dst_dir, clip_image)
    image_preprocess_util.resize_img(ori_img=clip_img, dst_img=resize_img, dst_w=resize, dst_h=resize)


print('Finished input image processing, deleting temp directory now.....')
image_preprocess_util.delete_file_folder(dog_temp_dir)
image_preprocess_util.delete_file_folder(cat_temp_dir)

print('Done！Please check images...')
end = time.time()

print('@ Total Time Spent: %.2f seconds' % (end - start))
