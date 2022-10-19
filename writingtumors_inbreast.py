# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 01:30:42 2022

@author: Talha
"""

import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from datetime import datetime 
import cv2
from PIL import Image
from keras import backend, optimizers


image_directory = 'G:/BreastCancerWorking/Datasets/INbreast/ClassesData_Augmented/Benign/'
#mask_directory = 'G:/BreastCancerWorking/Datasets/INbreast/processed_INbreast/All_Masks/'


SIZE = 512
image_dataset = []  #Many ways to handle data, you can use pandas. Here, we are using a list format.  
mask_dataset = []  #Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.

# images = os.listdir(image_directory)
# for i, image_name in enumerate(images):    #Remember enumerate method adds a counter and returns the enumerate object
#     if (image_name.split('.')[1] == 'jpg'):
#         #print(image_directory+image_name)
#         original = cv2.imread(image_directory+image_name, 1)
#         original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
#         blur_not = cv2.GaussianBlur(original,(5,5),0)
#         clahe = cv2.createCLAHE(clipLimit=5)
#         clahe_img = clahe.apply(blur_not)
#         #image = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB)
#         image = Image.fromarray(clahe_img)
#         image = image.resize((SIZE, SIZE))
        
#         image_dataset.append(np.array(image))

# #Iterate through all images in Uninfected folder, resize to 64 x 64
# #Then save into the same numpy array 'dataset' but with label 1

# masks = os.listdir(mask_directory)
# for i, image_name in enumerate(masks):
#     if (image_name.split('.')[1] == 'jpg'):
#         image = cv2.imread(mask_directory+image_name, 0)
#         image = Image.fromarray(image)
#         image = image.resize((SIZE, SIZE))
#         mask_dataset.append(np.array(image))


# #Normalize images
# image_dataset = np.array(image_dataset)/255.
# #D not normalize masks, just rescale to 0 to 1.
# mask_dataset = np.expand_dims((np.array(mask_dataset)),3) /255.

import os
os.chdir('G:/BreastCancerWorking/Python Codes/')
from unet_models import UNetplusplus, dice_coef, dice_coef_loss, jacard_coef

model_path = "G:/BreastCancerWorking/Python Codes/unetpp_checkpoints/UNetpp_250epochs_B_focal.hdf5"

input_shape=(512,512,1)

unetpp_model = UNetplusplus(input_shape,using_deep_supervision=False)

model = unetpp_model.load_weights(model_path)


model= unetpp_model


images = os.listdir(image_directory)
for i, image_name in enumerate(images):    #Remember enumerate method adds a counter and returns the enumerate object
    if (image_name.split('.')[1] == 'jpg'):
        #print(image_directory+image_name)
        original = cv2.imread(image_directory+image_name, 1)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        blur_not = cv2.GaussianBlur(original,(5,5),0)
        clahe = cv2.createCLAHE(clipLimit=5)
        clahe_img = clahe.apply(blur_not)
        #image = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB)
        image = Image.fromarray(clahe_img)
        image = image.resize((SIZE, SIZE))
        image = np.array(image)/255.
        test_img_input=np.expand_dims(image, 0)
        prediction = (model.predict(test_img_input)[0,:,:,0] > 0.5).astype(np.uint8)
        img  = prediction.astype('uint8') #Read as color (3 channels)
        kernel=np.ones((3,3),np.uint8)
        erode=cv2.erode(img,kernel,iterations=1)
        r=erode.astype('float64')*image


        #image_name=image_name.replace(".jpg","")
        #imagename = 'G:/BreastCancerWorking/Datasets/INbreast/Segmented_ClassesData_Augmented/Benign/'+image_name+'.jpg'
        #cv2.imwrite(imagename, 255*r)
        
import pandas as pd        
model_df = pd.DataFrame(model.summary()) 
with open('_unetpp_model_summary_df.csv', mode='w') as f:
    model_df.to_csv(f)
        