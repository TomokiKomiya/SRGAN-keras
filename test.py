#!/usr/bin/env python
#title           :test.py
#description     :to test the model
#author          :Tomoki
#date            :2020/05/09
#usage           :python test.py --options
#python_version  :3.7.3

from keras.models import Model
import matplotlib.pyplot as plt
import tensorflow as tf
import skimage.transform
from skimage import data, io, filters
import numpy as np
from numpy import array
import os
from keras.models import load_model
from scipy.misc import imresize
import argparse

import Utils, Utils_model
from Utils_model import VGG_LOSS

image_shape = (256, 256, 1)

def test_model(input_hig_res, model, number_of_images, output_dir):
    
    x_test_lr, x_test_hr = Utils.load_test_data_for_model(input_hig_res, 'jpg', number_of_images)

    x_test_lr = x_test_lr.reshape(x_test_lr.shape[0], x_test_lr.shape[1], x_test_lr.shape[2], 1)
    x_test_hr = x_test_hr.reshape(x_test_hr.shape[0], x_test_hr.shape[1], x_test_hr.shape[2], 1)

    Utils.plot_test_generated_images_for_model(output_dir, model, x_test_hr, x_test_lr)

def test_model_for_lr_images(input_low_res, model, number_of_images, output_dir):

    x_test_lr = Utils.load_test_data(input_low_res, 'jpg', number_of_images)
        
    x_test_lr = x_test_lr.reshape(x_test_lr.shape[0], x_test_lr.shape[1], x_test_lr.shape[2], 1)
    
    Utils.plot_test_generated_images(output_dir, model, x_test_lr)

if __name__== "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-ihr', '--input_hig_res', action='store', dest='input_hig_res', default='./data/sino-x-all-data/' ,
                    help='Path for input images Hig resolution')
                    
    parser.add_argument('-ilr', '--input_low_res', action='store', dest='input_low_res', default='./data_lr/sino-x-in-data/' ,
                    help='Path for input images Low resolution')
                    
    parser.add_argument('-o', '--output_dir', action='store', dest='output_dir', default='./output/x-result-sleep/' ,
                    help='Path for Output images')
    
    parser.add_argument('-m', '--model_dir', action='store', dest='model_dir', default='./model/sino-x-all-data/gen_model1000.h5' ,
                    help='Path for model')

    parser.add_argument('-n', '--number_of_images', action='store', dest='number_of_images', default=25 ,
                    help='Number of Images', type=int)
                    
    parser.add_argument('-t', '--test_type', action='store', dest='test_type', default='test_model',
                    help='Option to test model output or to test low resolution image')
    
    values = parser.parse_args()
    
    loss = VGG_LOSS(image_shape)  
    model = load_model(values.model_dir , custom_objects={'vgg_loss': loss.vgg_loss})
    
    if values.test_type == 'test_model':
        test_model(values.input_hig_res, model, values.number_of_images, values.output_dir)
        print("test_model")
    elif values.test_type == 'test_lr_images':
        print("test_lr_images")
        test_model_for_lr_images(values.input_low_res, model, values.number_of_images, values.output_dir)
        
    else:
        print("No such option")




