# -*- coding: utf-8 -*-
"""
Created on Thu May 12 12:27:40 2022

@author: Talha
"""
import os
import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify
import cv2
from PIL import Image
from keras import backend, optimizers


image_directory = 'G:/BreastCancerWorking/Datasets/INbreast/processed_INbreast/All_Images/'
mask_directory = 'G:/BreastCancerWorking/Datasets/INbreast/processed_INbreast/All_Masks/'

SIZE = 512
image_dataset = []  #Many ways to handle data, you can use pandas. Here, we are using a list format.  
mask_dataset = []  #Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.

images = os.listdir(image_directory)
for e, image_name in enumerate(images):    #Remember enumerate method adds a counter and returns the enumerate object
    if (image_name.split('.')[1] == 'jpg'):
        #print(image_directory+image_name)
        original = cv2.imread(image_directory+image_name, 1)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        
        patches_img = patchify(original, (256,256), step=256)
        for i in range(patches_img.shape[0]):
            for j in range(patches_img.shape[1]):
                single_patch_img = patches_img[i,j,:,:]
                blur_not = cv2.GaussianBlur(single_patch_img,(5,5),0)
                clahe = cv2.createCLAHE(clipLimit=5)
                single_patch_img = clahe.apply(blur_not)
                filename = 'G:/BreastCancerWorking/Datasets/INbreast/Augmented4_INbreast/256_Images/'+str(e)+'image_'+str(i)+str(j)+'.jpg'
                cv2.imwrite(filename, single_patch_img)

                image_dataset.append(np.array(original))

masks = os.listdir(mask_directory)

for k, image_name in enumerate(masks):    #Remember enumerate method adds a counter and returns the enumerate object
    if (image_name.split('.')[1] == 'jpg'):
        #print(image_directory+image_name)
        mask = cv2.imread(mask_directory+image_name, 0)
        
        patches_mask = patchify(mask, (256,256), step=256)
        for i in range(patches_mask.shape[0]):
            for j in range(patches_mask.shape[1]):
                single_patch_mask = patches_mask[i,j,:,:]
                cv2.imwrite('G:/BreastCancerWorking/Datasets/INbreast/Augmented4_INbreast/256_Masks/'+str(k)+'mask_'+str(i)+str(j)+'.jpg', single_patch_mask)
                single_patch_mask = single_patch_mask/255.
                mask_dataset.append(np.array(mask))
