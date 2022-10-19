# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 00:25:14 2022

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


image_directory = 'G:/BreastCancerWorking/Datasets/CBIS-DDSM/Class-wise-data/test_classes_data/Benign without callback/images/'
mask_directory = 'G:/BreastCancerWorking/Datasets/CBIS-DDSM/Class-wise-data/test_classes_data/Benign without callback/masks/'


SIZE = 512
image_dataset = []  #Many ways to handle data, you can use pandas. Here, we are using a list format.  
mask_dataset = []  #Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.

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
        
        image_dataset.append(np.array(image))

#Iterate through all images in Uninfected folder, resize to 64 x 64
#Then save into the same numpy array 'dataset' but with label 1

masks = os.listdir(mask_directory)
for i, image_name in enumerate(masks):
    if (image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(mask_directory+image_name, 0)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        mask_dataset.append(np.array(image))


#Normalize images
image_dataset = np.array(image_dataset)/255.
#D not normalize masks, just rescale to 0 to 1.
mask_dataset = np.expand_dims((np.array(mask_dataset)),3) /255.



import random
import numpy as np
#image_number = random.randint(0, len(X_train))
#plt.figure(figsize=(12, 6))
#plt.subplot(121)
#plt.imshow(np.reshape(X_train[image_number], (256, 256,3)), cmap='gray')
#plt.subplot(122)
#plt.imshow(np.reshape(y_train[image_number], (256, 256)), cmap='gray')
#plt.show()

#######################################
#Parameters for model

IMG_HEIGHT = 512
IMG_WIDTH  =512
IMG_CHANNELS = 1
num_labels = 1  #Binary
input_shape = (IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS)
# batch_size = 1

#FOCAL LOSS AND DICE METRIC
#Focal loss helps focus more on tough to segment classes.
#from focal_loss import binary_focal_loss 

###############################################################################


#Try various models: Unet, Attention_UNet, and Attention_ResUnet
#Rename original python file from 224_225_226_models.py to models.py
import os
os.chdir('G:/BreastCancerWorking/Python Codes/')
from unet_models import UNetplusplus, My_UNet, dice_coef, dice_coef_loss, jacard_coef

model_path = "G:/BreastCancerWorking/Python Codes/unetpp_cbis_checkpoints/model_.13-0.019437.hdf5"

unetpp_model = UNetplusplus(input_shape,using_deep_supervision=False)

model = unetpp_model.load_weights(model_path)


model= unetpp_model

import random
test_img_number = 30
test_img = image_dataset[test_img_number]
ground_truth=mask_dataset[test_img_number]

test_img_input=np.expand_dims(test_img, 0)
prediction = (model.predict(test_img_input)[0,:,:,0] > 0.5).astype(np.uint8)
img  = prediction.astype('uint8') #Read as color (3 channels)
kernel=np.ones((3,3),np.uint8)
erode=cv2.erode(img,kernel,iterations=1)

plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img, cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='gray')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(erode, cmap='gray')

plt.show()

#IoU for a single image
from tensorflow.keras.metrics import MeanIoU
n_classes = 2
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(ground_truth[:,:,0], prediction)
print("Mean IoU =", IOU_keras.result().numpy())
dice = dice_coef(ground_truth[:,:,0], erode.astype('float64'))
print("Mean Dice =", dice)


import pandas as pd

IoU_values = []
dice_values = []




for img in range(0, X_test.shape[0]):
    temp_img = image_dataset[img]
    ground_truth=mask_dataset[img]
    temp_img_input=np.expand_dims(temp_img, 0)
    prediction = (model.predict(temp_img_input)[0,:,:,0] > 0.5).astype(np.uint8)
    
    r=prediction.astype('float64')*temp_img

    IoU = MeanIoU(num_classes=n_classes)
    IoU.update_state(ground_truth[:,:,0], prediction)
    IoU = IoU.result().numpy()
    IoU_values.append(IoU)

    dice = dice_coef(ground_truth[:,:,0], prediction.astype('float64'))
    dice_values.append(dice)

    print('IoU for Image:',IoU)
    print('Dice for Image:',dice)
    


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
        #kernel=np.ones((3,3),np.uint8)
        #erode=cv2.erode(img,kernel,iterations=1)
        r=img.astype('float64')*image


        image_name=image_name.replace(".jpg","")
        imagename = 'G:/BreastCancerWorking/Datasets/CBIS-DDSM/Class-wise-data/test_classes_data/Benign without callback/processed_predictions/'+image_name+'.jpg'
        cv2.imwrite(imagename, 255*r)