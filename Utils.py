#!/usr/bin/env python
#title           :Utils.py
#description     :Have helper functions to process images and plot images
#author          :Tomoki
#date            :2020/05/09
#usage           :imported in other files
#python_version  :3.5.4

from keras.layers import Lambda
import tensorflow as tf
from skimage import data, io, filters
import numpy as np
from numpy import array
from numpy.random import randint
from scipy.misc import imresize
import os
import sys
from PIL import Image
from time import sleep
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# Better to use downscale factor as 4 (2)
downscale_factor = 2

# Subpixel Conv will upsample from (h, w, c) to (h/r, w/r, c/r^2)
def SubpixelConv2D(input_shape, scale=4):
    def subpixel_shape(input_shape):
        dims = [input_shape[0],input_shape[1] * scale,input_shape[2] * scale,int(input_shape[3] / (scale ** 2))]
        output_shape = tuple(dims)
        return output_shape
    
    def subpixel(x):
        return tf.depth_to_space(x, scale)
        
    return Lambda(subpixel, output_shape=subpixel_shape)
    
# Takes list of images and provide HR images in form of numpy array
def hr_images(images):
    images_hr = array(images)
    return images_hr

# Takes list of images and provide LR images in form of numpy array
def lr_images(images_real , downscale):
    
    images = []
    for img in  range(len(images_real)):
        images.append(imresize(images_real[img], [images_real[img].shape[0]//downscale,images_real[img].shape[1]//downscale], interp='nearest', mode=None))
    images_lr = array(images)
    return images_lr
    
def normalize(input_data):

    return (input_data.astype(np.float32) - 127.5)/127.5 
    
def denormalize(input_data):
    input_data = (input_data + 1) * 127.5
    return input_data.astype(np.uint8)
   
 
def load_path(path):
    directories = []
    if os.path.isdir(path):
        directories.append(path)
    for elem in os.listdir(path):
        if os.path.isdir(os.path.join(path,elem)):
            directories = directories + load_path(os.path.join(path,elem))
            directories.append(os.path.join(path,elem))
    return directories
    
def load_data_from_dirs(dirs, ext):
    files = []
    file_names = []
    count = 0
    for d in dirs:
        for f in os.listdir(d):
            if f.endswith(ext):
                image = plt.imread(os.path.join(d,f))
                files.append(image)
                file_names.append(os.path.join(d,f))
                count = count + 1
    return files     

def load_data(directory, ext):

    files = load_data_from_dirs(load_path(directory), ext)
    return files
    
def load_training_data(directory, ext, number_of_images = 1000, train_test_ratio = 0.8):

    number_of_train_images = int(number_of_images * train_test_ratio)
    
    files = load_data_from_dirs(load_path(directory), ext)
    
    if len(files) < number_of_images:
        print("Number of image files are less then you specified")
        print("Please reduce number of images to %d" % len(files))
        sys.exit()
        
    test_array = array(files)
    if len(test_array.shape) < 3:
        print("Images are of not same shape")
        print("Please provide same shape images")
        sys.exit()
    
    x_train = files[:number_of_train_images]
    x_test = files[number_of_train_images:number_of_images]

    x_train_hr = hr_images(x_train)
    x_train_hr = normalize(x_train_hr)
    
    x_train_lr = lr_images(x_train, downscale_factor)
    x_train_lr = normalize(x_train_lr)
    
    x_test_hr = hr_images(x_test)
    x_test_hr = normalize(x_test_hr)
    
    x_test_lr = lr_images(x_test, downscale_factor)
    x_test_lr = normalize(x_test_lr)
    
    return x_train_lr, x_train_hr, x_test_lr, x_test_hr


def load_test_data_for_model(directory, ext, number_of_images = 100):

    files = load_data_from_dirs(load_path(directory), ext)
    
    if len(files) < number_of_images:
        print("Number of image files are less then you specified")
        print("Please reduce number of images to %d" % len(files))
        sys.exit()
        
    x_test_hr = hr_images(files)
    x_test_hr = normalize(x_test_hr)
    
    x_test_lr = lr_images(files, downscale_factor)
    x_test_lr = normalize(x_test_lr)
    
    return x_test_lr, x_test_hr
    
def load_test_data(directory, ext, number_of_images = 100):

    files = load_data_from_dirs(load_path(directory), ext)
    
    if len(files) < number_of_images:
        print("Number of image files are less then you specified")
        print("Please reduce number of images to %d" % len(files))
        sys.exit()
        
    x_test_lr = lr_images(files, downscale_factor)
    x_test_lr = normalize(x_test_lr)
    
    return x_test_lr
    
# While training save generated image(in form LR, SR, HR)
# Save only one image as sample  
def plot_generated_images(output_dir, epoch, generator, x_test_hr, x_test_lr , dim=(1, 3), figsize=(15, 5)):
    
    examples = x_test_hr.shape[0]

    value = randint(0, examples)
    image_batch_hr = denormalize(x_test_hr)
    image_batch_hr = image_batch_hr.reshape(image_batch_hr.shape[0], image_batch_hr.shape[1], image_batch_hr.shape[2], 1)

    image_batch_lr = x_test_lr
    image_batch_lr = image_batch_lr.reshape(image_batch_lr.shape[0], image_batch_lr.shape[1], image_batch_lr.shape[2], 1)

    gen_img = generator.predict(image_batch_lr)

    generated_image = denormalize(gen_img)
    generated_image = generated_image.reshape(generated_image.shape[0], generated_image.shape[1], generated_image.shape[2], 1)

    image_batch_lr = denormalize(image_batch_lr)

    image_batch_hr = image_batch_hr.reshape(image_batch_hr.shape[0], image_batch_hr.shape[1], image_batch_hr.shape[2])
    image_batch_lr = image_batch_lr.reshape(image_batch_lr.shape[0], image_batch_lr.shape[1], image_batch_lr.shape[2])
    generated_image = generated_image.reshape(generated_image.shape[0], generated_image.shape[1], generated_image.shape[2])

    plt.figure(figsize=figsize)
    
    plt.subplot(dim[0], dim[1], 1)
    plt.imshow(image_batch_lr[value], cmap = "gray", interpolation='None')
    plt.axis('off')
        
    plt.subplot(dim[0], dim[1], 2)
    plt.imshow(generated_image[value], cmap = "gray", interpolation='None')
    plt.axis('off')
    
    plt.subplot(dim[0], dim[1], 3)
    plt.imshow(image_batch_hr[value], cmap = "gray", interpolation='None')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir + 'generated_image_%d.png' % epoch)
    
    # plt.show()

    print("Image Saved Successfully")
    
# Plots and save generated images(in form LR, SR, HR) from model to test the model 
# Save output for all images given for testing  
def plot_test_generated_images_for_model(output_dir, generator, x_test_hr, x_test_lr , dim=(1, 3), figsize=(15, 5)):
    
    examples = x_test_hr.shape[0]
    image_batch_hr = denormalize(x_test_hr)
    image_batch_lr = x_test_lr
    gen_img = generator.predict(image_batch_lr)
    generated_image = denormalize(gen_img)
    image_batch_lr = denormalize(image_batch_lr)
    
    for index in range(examples):
        print("begin")

        # image_batch_hr_res = image_batch_hr[index].reshape(image_batch_hr.shape[1], image_batch_hr.shape[2])
        # image_batch_lr_res = image_batch_lr[index].reshape(image_batch_lr.shape[1], image_batch_lr.shape[2])
        generated_image_res = generated_image[index].reshape(generated_image.shape[1], generated_image.shape[2])

        # print(image_batch_hr_res.shape)
        # print(image_batch_lr_res.shape)
        # print(generated_image_res.shape)
        # print(type(generated_image_res))
        # print("==========================")
        
        pil_img_gray = Image.fromarray(generated_image_res)
        pil_img_gray.save(output_dir + 'result_image_%d.png' % index)

        sleep(0.01)
        print("end")
        # plt.figure(figsize=figsize)

        # plt.subplot(dim[0], dim[1], 1)
        # plt.imshow(image_batch_lr_res, cmap = "gray", interpolation='None')
        # plt.axis('off')
        
        # plt.subplot(dim[0], dim[1], 2)
        # plt.imshow(generated_image_res, cmap = "gray", interpolation='None')
        # plt.axis('off')
    
        # plt.subplot(dim[0], dim[1], 3)
        # plt.imshow(image_batch_hr_res, cmap = "gray", interpolation='None')
        # plt.axis('off')
    
        # plt.tight_layout()
        # plt.savefig(output_dir + 'test_generated_image_%d.png' % index)
    
        #plt.show()

    print("Image Saved Successfully")

# Takes LR images and save respective HR images
def plot_test_generated_images(output_dir, generator, x_test_lr, figsize=(5, 5)):
    
    examples = x_test_lr.shape[0]
    image_batch_lr = denormalize(x_test_lr)
    gen_img = generator.predict(image_batch_lr)
    generated_image = denormalize(gen_img)
    
    for index in range(examples):
        
        generated_image_res = generated_image[index].reshape(generated_image.shape[1], generated_image.shape[2])

        #plt.figure(figsize=figsize)
    
        plt.imshow(generated_image_res, cmap = "gray", interpolation='None')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir + 'high_res_result_image_%d.png' % index)
    
        #plt.show()
    
    print("Image Saved Successfully")





