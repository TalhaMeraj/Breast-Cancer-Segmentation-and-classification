# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 03:08:18 2022

@author: Talha
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 20 02:30:20 2022

@author: Talha
"""

import albumentations as A
import cv2
from matplotlib import pyplot as plt
import random


#image = cv2.imread('G:/BreastCancerWorking/Datasets/INbreast/processed_INbreast/Train Images/27829134.jpg',1)
#mask = cv2.imread('G:/BreastCancerWorking/Datasets/INbreast/processed_INbreast/Train Masks/27829134.jpg',0)
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


from albumentations import (
    Compose, RandomBrightness, JpegCompression, HueSaturationValue, RandomContrast, HorizontalFlip,
    Rotate
)

###############################
###Reading images and Masks####


import os



image_directory = 'G:/BreastCancerWorking/Datasets/INbreast/Classes_Data/Malignant/'
#mask_directory = 'G:/BreastCancerWorking/Datasets/INbreast/processed_INbreast/All Masks/'


#SIZE = 256
image_dataset = []  #Many ways to handle data, you can use pandas. Here, we are using a list format.  
mask_dataset = []  #Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.

images = os.listdir(image_directory)
for i, image_name in enumerate(images):    #Remember enumerate method adds a counter and returns the enumerate object
    if (image_name.split('.')[1] == 'jpg'):
        #print(image_directory+image_name)
        image = cv2.imread(image_directory+image_name, 1)
        original_height, original_width = image.shape[:2]

        #mask = cv2.imread(mask_directory+image_name, 0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image_name=image_name.replace(".jpg","")
        imagename = 'G:/BreastCancerWorking/Datasets/INbreast/ClassesData_Augmented/Malignant/'+image_name+'_original'+'.jpg'
        cv2.imwrite(imagename, image)
        #maskname = 'G:/BreastCancerWorking/Datasets/INbreast/Augmented3_INbreast/All Masks/'+image_name+'_original'+'.jpg'
        #cv2.imwrite(maskname, mask)
        
        # ## Light non-destructive augmentations.
        aug1 = A.Compose([
            A.VerticalFlip(p=0.5),              
            A.RandomRotate90(p=0.5)]
        )
        random.seed(7)
        augmented_1 = aug1(image=image)
        image_light = augmented_1['image']
        #mask_light = augmented_1['mask']
        
        image_name=image_name.replace(".jpg","")
        imagename = 'G:/BreastCancerWorking/Datasets/INbreast/ClassesData_Augmented/Malignant/'+image_name+'_light'+'.jpg'
        cv2.imwrite(imagename, image_light)
       # maskname = 'G:/BreastCancerWorking/Datasets/INbreast/Augmented3_INbreast/All Masks/'+image_name+'_light'+'.jpg'
        #cv2.imwrite(maskname, mask_light)

        
        
        ## Adding non rigid transformations and RandomSizedCrop
        # aug2 = A.Compose([
        #     A.OneOf([
        #         A.RandomSizedCrop(min_max_height=(50, 101), height=original_height, width=original_width, p=0.5),
        #         A.PadIfNeeded(min_height=original_height, min_width=original_width, p=0.5)
        #     ],p=1),
        #     A.VerticalFlip(p=0.5),
        #     A.RandomRotate90(p=0.5),
        #     A.OneOf([
        #         A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        #         A.GridDistortion(p=0.5),
        #         A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1),
        #     ], p=0.8)])
        
        # random.seed(11)
        # augmented_2 = aug2(image=image, mask=mask)
        
        # image_medium = augmented_2['image']
        # mask_medium = augmented_2['mask']
        
        # image_name=image_name.replace(".jpg","")
        # imagename = 'G:/BreastCancerWorking/Datasets/INbreast/Augmented3_INbreast/Test Images/'+image_name+'_medium'+'.jpg'
        # cv2.imwrite(imagename, image_medium)
        # maskname = 'G:/BreastCancerWorking/Datasets/INbreast/Augmented3_INbreast/Test Masks/'+image_name+'_medium'+'.jpg'
        # cv2.imwrite(maskname, mask_medium)
        
        
        ## Grid Distortion        
        aug3 = A.GridDistortion(p=1)
        
        random.seed(7)
        augmented_3 = aug3(image=image)
        
        image_grid = augmented_3['image']
        #mask_grid = augmented_3['mask']
        
        image_name=image_name.replace(".jpg","")
        imagename = 'G:/BreastCancerWorking/Datasets/INbreast/ClassesData_Augmented/Malignant/'+image_name+'_grid'+'.jpg'
        cv2.imwrite(imagename, image_grid)
        #maskname = 'G:/BreastCancerWorking/Datasets/INbreast/Augmented3_INbreast/All Masks/'+image_name+'_grid'+'.jpg'
        #cv2.imwrite(maskname, mask_grid)
        
        
        
        # Elastic  
        
        # aug4 = A.ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)

        # random.seed(7)
        # augmented_4 = aug4(image=image, mask=mask)
        
        # image_elastic = augmented_4['image']
        # mask_elastic = augmented_4['mask']
        
        # image_name=image_name.replace(".jpg","")
        # imagename = 'G:/BreastCancerWorking/Datasets/INbreast/Augmented3_INbreast/Test Images/'+image_name+'_elastic'+'.jpg'
        # cv2.imwrite(imagename, image_elastic)
        # maskname = 'G:/BreastCancerWorking/Datasets/INbreast/Augmented3_INbreast/Test Masks/'+image_name+'_elastic'+'.jpg'
        # cv2.imwrite(maskname, mask_elastic)
        
        
        # Horizantal Flip
        
        aug5 = A.HorizontalFlip(p=1)

        augmented_5 = aug5(image=image)
        
        image_h_flipped = augmented_5['image']
        #mask_h_flipped = augmented_5['mask']
        
        image_name=image_name.replace(".jpg","")
        imagename = 'G:/BreastCancerWorking/Datasets/INbreast/ClassesData_Augmented/Malignant/'+image_name+'_h_flipped'+'.jpg'
        cv2.imwrite(imagename, image_h_flipped)
        #maskname = 'G:/BreastCancerWorking/Datasets/INbreast/Augmented3_INbreast/All Masks/'+image_name+'_h_flipped'+'.jpg'
        #cv2.imwrite(maskname, mask_h_flipped)
        
        
        # Vertical Flip
        
        aug6 = A.VerticalFlip(p=1)
        
        augmented_6 = aug6(image=image)
        
        image_v_flipped = augmented_6['image']
        #mask_v_flipped = augmented_6['mask']
        
        image_name=image_name.replace(".jpg","")
        imagename = 'G:/BreastCancerWorking/Datasets/INbreast/ClassesData_Augmented/Malignant/'+image_name+'_v_flipped'+'.jpg'
        cv2.imwrite(imagename, image_v_flipped)
        #maskname = 'G:/BreastCancerWorking/Datasets/INbreast/Augmented3_INbreast/All Masks/'+image_name+'_v_flipped'+'.jpg'
        #cv2.imwrite(maskname, mask_v_flipped)
        
        
        # Transpose
        
        aug7 = A.Transpose(p=1)

        augmented_7 = aug7(image=image)
        
        image_transposed = augmented_7['image']
        #mask_transposed = augmented_7['mask']

        image_name=image_name.replace(".jpg","")
        imagename = 'G:/BreastCancerWorking/Datasets/INbreast/ClassesData_Augmented/Malignant/'+image_name+'_transposed1'+'.jpg'
        cv2.imwrite(imagename, image_transposed)
        #maskname = 'G:/BreastCancerWorking/Datasets/INbreast/Augmented3_INbreast/All Masks/'+image_name+'_transposed1'+'.jpg'
        #cv2.imwrite(maskname, mask_transposed)
        
        #image = image.resize((SIZE, SIZE))
        
        #image_dataset.append(np.array(image))
        
        # Random Size Cropping
        
        # aug8 = A.Compose([
        # A.OneOf([
        #         A.RandomSizedCrop(min_max_height=(50, 101), height=original_height, width=original_width, p=0.5),
        #         A.PadIfNeeded(min_height=original_height, min_width=original_width, p=0.5)
        #     ], p=1),    
        #     A.VerticalFlip(p=0.5),              
        #     A.RandomRotate90(p=0.5),
        #     A.OneOf([
        #         A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
        #         A.GridDistortion(p=0.5),
        #         A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1)                  
        #         ], p=0.8),
        # A.CLAHE(p=0.8),
        # A.RandomBrightnessContrast(p=0.8),    
        # A.RandomGamma(p=0.8)])
        
        # random.seed(11)
        # augmented_8 = aug8(image=image, mask=mask)
        
        # image_heavy = augmented_8['image']
        # mask_heavy = augmented_8['mask']
        # image_name=image_name.replace(".jpg","")
        # imagename = 'G:/BreastCancerWorking/Datasets/INbreast/Augmented3_INbreast/All Images/'+image_name+'_heavy'+'.jpg'
        # cv2.imwrite(imagename, image_heavy)
        # maskname = 'G:/BreastCancerWorking/Datasets/INbreast/Augmented3_INbreast/All Masks/'+image_name+'_heavy'+'.jpg'
        # cv2.imwrite(maskname, mask_heavy)
        
        
        
        # Transpose
        
        aug9 = A.RandomRotate90(p=1)
        
        augmented_9 = aug9(image=image)
        
        image_rot90 = augmented_9['image']
       # mask_rot90 = augmented_9['mask']
        
        image_name=image_name.replace(".jpg","")
        imagename = 'G:/BreastCancerWorking/Datasets/INbreast/ClassesData_Augmented/Malignant/'+image_name+'_transposed2'+'.jpg'
        cv2.imwrite(imagename, image_rot90)
        #maskname = 'G:/BreastCancerWorking/Datasets/INbreast/Augmented3_INbreast/All Masks/'+image_name+'_transposed2'+'.jpg'
        #cv2.imwrite(maskname, mask_rot90)


############################










def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 18
    
    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)
        
        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)
        
        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)
        
        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)


print(image.shape, mask.shape)
original_height, original_width = image.shape[:2]
visualize(image, mask)


## Light non-destructive augmentations.
aug = A.Compose([
    A.VerticalFlip(p=0.5),              
    A.RandomRotate90(p=0.5)]
)

random.seed(7)
augmented = aug(image=image, mask=mask)

image_light = augmented['image']
mask_light = augmented['mask']

visualize(image_light, mask_light, original_image=image, original_mask=mask)


## Adding non rigid transformations and RandomSizedCrop
aug = A.Compose([
    A.OneOf([
        A.RandomSizedCrop(min_max_height=(50, 101), height=original_height, width=original_width, p=0.5),
        A.PadIfNeeded(min_height=original_height, min_width=original_width, p=0.5)
    ],p=1),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.OneOf([
        A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        A.GridDistortion(p=0.5),
        A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1),
    ], p=0.8)])

random.seed(11)
augmented = aug(image=image, mask=mask)

image_medium = augmented['image']
mask_medium = augmented['mask']

visualize(image_medium, mask_medium, original_image=image, original_mask=mask)

## Grid Distortion

aug = A.GridDistortion(p=1)

random.seed(7)
augmented = aug(image=image, mask=mask)

image_grid = augmented['image']
mask_grid = augmented['mask']

visualize(image_grid, mask_grid, original_image=image, original_mask=mask)

## Elastic Distortion

aug = A.ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)

random.seed(7)
augmented = aug(image=image, mask=mask)

image_elastic = augmented['image']
mask_elastic = augmented['mask']

visualize(image_elastic, mask_elastic, original_image=image, original_mask=mask)




