# -*- coding: utf-8 -*-
# @Time    : 2018/1/18 14:39
# @File    : input_images.py
# @Author  : NusLuoKe

import os
from image_preprocess_util import resize_img, rename_files, generate_image, clip_resize_img, delete_file_folder

# followings are my default testing directories, when use this script, comment these 4 lines.
image_dir = 'T:/Cat_Dog/cat'
resized_img_dir = 'T:/Test/r_cats'
clipResized_img_dir = 'T:/Test/cr_cats'
augmented_img_dir = 'T:/Cat_Dog/cats'

# compress the image.
# eg: ori: 100*60 -> set resize_img_capped_len = 50 -> des: 50*30
resize_img_capped_len = 150

# crop the image to the specified size.
clip_resize_img_w = 128
clip_resize_img_h = 128

# number of new generate images from per image
gen_num = 1

# prefix of the name of augment images
prefix = 'cat'

if not os.path.isdir(resized_img_dir):
    os.makedirs(resized_img_dir)

if not os.path.isdir(clipResized_img_dir):
    os.makedirs(clipResized_img_dir)

if not os.path.isdir(augmented_img_dir):
    os.makedirs(augmented_img_dir)

for image in os.listdir(image_dir):
    if not os.path.isdir(image):
        ori_img = os.path.join(image_dir, image)
        dst_img = os.path.join(resized_img_dir, image)
        dst_w = resize_img_capped_len
        dst_h = resize_img_capped_len
        save_q = 100
        resize_img(ori_img=ori_img, dst_img=dst_img, dst_w=dst_w, dst_h=dst_h, save_q=save_q)

for resi_image in os.listdir(resized_img_dir):
    if not os.path.isdir(resi_image):
        ori_img = os.path.join(resized_img_dir, resi_image)
        dst_img = os.path.join(clipResized_img_dir, resi_image)
        dst_w = clip_resize_img_w
        dst_h = clip_resize_img_w
        save_q = 100
        clip_resize_img(ori_img=ori_img, dst_img=dst_img, dst_w=dst_w, dst_h=dst_h, save_q=save_q)

generate_image(img_dir=clipResized_img_dir, save_dir=augmented_img_dir, prefix=prefix, gen_num=gen_num)
rename_files(file_dir=augmented_img_dir, new_prefix=prefix)

delete_file_folder(resized_img_dir)
delete_file_folder(clipResized_img_dir)
