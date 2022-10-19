# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 19:40:47 2022

@author: Talha
"""


import matplotlib.pyplot as plt
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Input, Dense, Dropout
import numpy as np

import cv2
image_dir='G:/BreastCancerWorking/Datasets/CBIS-DDSM'
test_image = image_dir+"/All Images/442_RIGHT_MLO.png"
img = image.load_img(test_image, target_size=(512, 512,1))
img_arr = image.img_to_array(img)
gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)

img_arr = np.expand_dims(gray, axis=0)

img_arr /= 255.


import os
os.chdir('G:/BreastCancerWorking/Python Codes/')

from unet_models import UNetplusplus, My_UNet, dice_coef, dice_coef_loss, jacard_coef
model_path = "G:/BreastCancerWorking/Python Codes/unetpp_cbis_checkpoints/model_.13-0.019437.hdf5"

input_shape=(512,512,1)

model_1 = UNetplusplus(input_shape,using_deep_supervision=False)

model = model_1.load_weights(model_path)

activations = (model_1.predict(img_arr)[0,:,:,0] > 0.5).astype(np.uint8)


import matplotlib.pyplot as plt


# These are names of layers, the outputs of which we are going to visualize.
layer_names = ['conv2d_1', 'activation_1', 'conv_7b', 'conv_7b_ac', 'final_conv', 'sigmoid']
activ_list = [activations[0], activations[2], activations[10], activations[12], activations[17], activations[19]]


fig = plt.figure(figsize=(22, 3))
for img in range(30):
    ax = fig.add_subplot(2, 15, img+1)
    ax = plt.imshow(activations[60][ gray.astype('int')], cmap='plasma')
    plt.xticks([])
    plt.yticks([])
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    
    
fig = plt.figure(figsize=(22, 6))
for img in range(60):
    ax = fig.add_subplot(4, 15, img+1)
    ax = plt.imshow(activations[17][gray.astype('int')], cmap='plasma')
    plt.xticks([])
    plt.yticks([])
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    
    