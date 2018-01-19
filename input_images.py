# -*- coding: utf-8 -*-
# @Time    : 2018/1/18 14:39
# @File    : input_images.py
# @Author  : NusLuoKe

import os
from image_preprocess import resize_img, rename_files, generate_image, clip_resize_img

image_dir = 'T:/data_augmentation_demo/original_images'
resized_img_dir = 'T:/data_augmentation_demo/resized_images'
clipResized_img_dir = 'T:/data_augmentation_demo/clipResized_img_dir_images'

if not os.path.isdir(resized_img_dir):
    os.makedirs(resized_img_dir)

if not os.path.isdir(clipResized_img_dir):
    os.makedirs(clipResized_img_dir)

for image in os.listdir(image_dir):
    if not os.path.isdir(image):
        ori_img = os.path.join(image_dir, image)
        dst_img = os.path.join(resized_img_dir, image)
        dst_w = 128
        dst_h = 128
        save_q = 75
        resize_img(ori_img=ori_img, dst_img=dst_img, dst_w=dst_w, dst_h=dst_h, save_q=save_q)

for resi_image in os.listdir(resized_img_dir):
    if not os.path.isdir(resi_image):
        ori_img = os.path.join(resized_img_dir, resi_image)
        dst_img = os.path.join(clipResized_img_dir, resi_image)
        dst_w = 128
        dst_h = 128
        save_q = 75
        clip_resize_img(ori_img=ori_img, dst_img=dst_img, dst_w=dst_w, dst_h=dst_h, save_q=save_q)

augmented_img_dir = 'T:/data_augmentation_demo/augmented_images'
if not os.path.isdir(augmented_img_dir):
    os.makedirs(augmented_img_dir)
generate_image(img_dir=clipResized_img_dir, save_dir=augmented_img_dir, prefix='pet', gen_num=5)
rename_files(file_dir=augmented_img_dir, new_prefix='pet')
