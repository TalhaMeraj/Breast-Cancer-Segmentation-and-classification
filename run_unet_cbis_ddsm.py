# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 03:59:50 2022

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


image_directory = 'G:/BreastCancerWorking/Datasets/CBIS-DDSM//test-images/'
mask_directory = 'G:/BreastCancerWorking/Datasets/CBIS-DDSM//test-masks/'


SIZE = 512
image_dataset = []  #Many ways to handle data, you can use pandas. Here, we are using a list format.  
mask_dataset = []  #Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.

images = os.listdir(image_directory)
for i, image_name in enumerate(images):    #Remember enumerate method adds a counter and returns the enumerate object
    if (image_name.split('.')[1] == 'png'):
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
    if (image_name.split('.')[1] == 'png'):
        image = cv2.imread(mask_directory+image_name, 0)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        mask_dataset.append(np.array(image))


#Normalize images
image_dataset = np.array(image_dataset)/255.
#D not normalize masks, just rescale to 0 to 1.
mask_dataset = np.expand_dims((np.array(mask_dataset)),3) /255.

#image_dataset = image_dataset.astype(np.float32)

#mask_dataset = mask_dataset.astype(np.float32)



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size = 0.20, random_state = 0)

#Sanity check, view few mages
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

IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH  = X_train.shape[2]
IMG_CHANNELS = 1
num_labels = 1  #Binary
input_shape = (IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS)
batch_size = 1

#FOCAL LOSS AND DICE METRIC
#Focal loss helps focus more on tough to segment classes.
#from focal_loss import binary_focal_loss 

###############################################################################


#Try various models: Unet, Attention_UNet, and Attention_ResUnet
#Rename original python file from 224_225_226_models.py to models.py
import os
os.chdir('G:/BreastCancerWorking/Python Codes/')
#os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/bin") 

from unet_models import My_UNet, Attention_ResUNet, UNet, Attention_UNet, dice_coef, dice_coef_loss, jacard_coef
'''
UNet
'''
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard, EarlyStopping

unet_model = UNet(input_shape)

unet_model.compile(optimizer=Adam(learning_rate = 1e-3), loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy', jacard_coef,dice_coef])

csv_logger = CSVLogger('training.log')

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)

mc = [ModelCheckpoint(filepath = 'G:/BreastCancerWorking/Python Codes/unetenhanced_checkpoints/model_.{epoch:02d}-{val_loss:.6f}.hdf5',verbose=1,monitor='val_loss',  save_best_only=True, save_weights_only = True,mode='min'), csv_logger]

unet_history = unet_model.fit(X_train, y_train, verbose=1,callbacks=[es,mc],batch_size = 1, validation_data=(X_test, y_test ), shuffle=False,epochs=100)



stop1 = datetime.now()
#Execution time of the model 
execution_time_Unet = stop1-start1
print("UNet execution time is: ", execution_time_Unet)

unet_model.save('G:/BreastCancerWorking/Python Codes/Unetpatches_checkpoints/_UNet_100epochs_B_focal.hdf5')


# 1st resume of checkpoints
# Recover the iterations from the model and convert to epochs
# model_path = 'G:/python_for_microscopists-master/Inbreast/INBreastResults/dilatedUnet/model_.44-0.048248.hdf5'
# dilated_unet_model.load_weights(model_path)
# dilated_unet_model.compile(optimizer=Adam(lr = 1e-2), loss=BinaryFocalLoss(gamma=2), metrics=['accuracy', jacard_coef,dice_coef])

#initial_epoch = dilated_unet_model.optimizer.iterations.numpy() 
csv_logger = CSVLogger('training.log')

input_shape=(512,512,3)
AttentionUnet_model = Attention_UNet(input_shape)

AttentionUnet_model.compile(optimizer=Adam(learning_rate = 1e-2), loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy', jacard_coef,dice_coef])

#csv_logger = CSVLogger('training.log')


callbacks = [ModelCheckpoint(filepath = 'G:/BreastCancerWorking/Python Codes/AttentionUnet_checkpoints/model_.{epoch:02d}-{dice_coef:.6f}.hdf5',verbose=1, save_best_only=True, save_weights_only = True), csv_logger]
AttentionUnet_history=AttentionUnet_model.fit(X_train, y_train, verbose=1,callbacks=callbacks,batch_size = 1,validation_data=(X_test, y_test ), epochs=50, initial_epoch=1)

AttentionUnet_model.save('G:/BreastCancerWorking/Python Codes/Unet_checkpoints/Attention_UNet_50epochs_B_focal.hdf5')

# load model
model_path = "G:/BreastCancerWorking/Python Codes/AttentionUnet_checkpoints/Attention_UNet_50epochs_B_focal.hdf5"

att_unet_model = Attention_UNet(input_shape)

model = att_unet_model.load_weights(model_path)




csv_logger = CSVLogger('training.log')


Attention_ResUNet_model = Attention_ResUNet(input_shape)

Attention_ResUNet_model.compile(optimizer=Adam(learning_rate = 1e-2), loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy', jacard_coef,dice_coef])

#csv_logger = CSVLogger('training.log')


callbacks = [ModelCheckpoint(filepath = 'G:/BreastCancerWorking/Python Codes/Attention_ResUnet_checkpoints/model_.{epoch:02d}-{dice_coef:.6f}.hdf5',verbose=1, save_best_only=True, save_weights_only = True), csv_logger]
Attention_ResUNet_history=Attention_ResUNet_model.fit(X_train, y_train, verbose=1,callbacks=callbacks,batch_size = 1,validation_data=(X_test, y_test ), epochs=50)

Attention_ResUNet_model.save('G:/BreastCancerWorking/Python Codes/AttentionUnet_checkpoints/AttentionResidual_UNet_50epochs_B_focal.hdf5')


## My_Unet

from unet_models import UNetplusplus, dice_coef, dice_coef_loss, jacard_coef

my_unet_model = My_UNet(input_shape)

my_unet_model.compile(optimizer=Adam(learning_rate = 1e-2), loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=['accuracy', jacard_coef,dice_coef])

csv_logger = CSVLogger('training.log')

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

mc = [ModelCheckpoint(filepath = 'G:/BreastCancerWorking/Python Codes/my_unet_checkpoints/model_.{epoch:02d}-{val_loss:.6f}.hdf5',verbose=1,monitor='val_loss',  save_best_only=True, save_weights_only = True,mode='min'), csv_logger]

my_unet_model_history = my_unet_model.fit(X_train, y_train, verbose=1,callbacks=[es,mc],batch_size = 4, validation_data=(X_test, y_test ), shuffle=False,epochs=100)

my_unet_model.save('G:/BreastCancerWorking/Python Codes/256_unetenhanced_checkpoints/My_UNet_11epochs_B_focal.hdf5')



model_path = "G:/BreastCancerWorking/Python Codes/unetpp_cbis_checkpoints/UNetpp_110epochs_earlystopping72.36_B_focal.hdf5"

unetpp_model = UNetplusplus(input_shape,using_deep_supervision=False)

model = unetpp_model.load_weights(model_path)


## UNet++
unetpp_model = UNetplusplus(input_shape,using_deep_supervision=False)
unetpp_model.compile(optimizer=Adam(learning_rate = 1e-2), loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=['accuracy', jacard_coef,dice_coef])

csv_logger = CSVLogger('training.log')

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)

mc = [ModelCheckpoint(filepath = 'G:/BreastCancerWorking/Python Codes/unetpp_cbis_checkpoints/model_.{epoch:02d}-{val_loss:.6f}.hdf5',verbose=1,monitor='val_loss',  save_best_only=True, save_weights_only = True,mode='min'), csv_logger]

unetpp_model_history = unetpp_model.fit(X_train, y_train, verbose=1,callbacks=[es,mc],batch_size = 1, validation_data=(X_test, y_test ), shuffle=False,epochs=50)

unetpp_model.save('G:/BreastCancerWorking/Python Codes/unetpp_cbis_checkpoints/UNetpp_110epochs_earlystopping72.36_B_focal.hdf5')




#model.load_weights(model_path)

model= unetpp_model

import random
test_img_number = 30
test_img = X_test[test_img_number]
ground_truth=y_test[test_img_number]

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


#Calculate IoU for all test images and average
 
import pandas as pd

IoU_values = []
dice_values = []

for img in range(0, image_dataset.shape[0]):
    temp_img = image_dataset[img]
    ground_truth=mask_dataset[img]
    temp_img_input=np.expand_dims(temp_img, 0)
    prediction = (model.predict(temp_img_input)[0,:,:,0] > 0.5).astype(np.uint8)
    
    IoU = MeanIoU(num_classes=n_classes)
    IoU.update_state(ground_truth[:,:,0], prediction)
    IoU = IoU.result().numpy()
    IoU_values.append(IoU)

    dice = dice_coef(ground_truth[:,:,0], prediction.astype('float64'))
    dice_values.append(dice)

    print('IoU for Image:',IoU)
    print('Dice for Image:',dice)
    


df_dice = pd.DataFrame(dice_values,columns=["dice"])
df_dice = df_dice[df_dice.dice != 1.0]
mean_dice = df_dice.mean().values
print("Mean Dice is:", mean_dice)
df = pd.DataFrame(IoU_values, columns=["IoU"])
df = df[df.IoU != 1.0]    
mean_IoU = df.mean().values
print("Mean IoU is: ", mean_IoU)    










## cbis-ddsm testing

from tensorflow.keras.metrics import MeanIoU

import pandas as pd


model=att_unet_model
IoU_values = []
dice_values = []
n_classes=2
for img in range(0, image_dataset.shape[0]):
    temp_img = image_dataset[img]
    ground_truth=mask_dataset[img]
    temp_img_input=np.expand_dims(temp_img, 0)
    prediction = (model.predict(temp_img_input)[0,:,:,0] > 0.5).astype(np.uint8)
    
    IoU = MeanIoU(num_classes=n_classes)
    IoU.update_state(ground_truth[:,:,0], prediction)
    IoU = IoU.result().numpy()
    IoU_values.append(IoU)

    dice = dice_coef(ground_truth[:,:,0], prediction.astype('float64'))
    dice_values.append(dice)

    print('IoU for Image:',IoU)
    print('Dice for Image:',dice)
    


df_dice = pd.DataFrame(dice_values,columns=["dice"])
df_dice = df_dice[df_dice.dice != 1.0]
mean_dice = df_dice.mean().values
print("Mean Dice is:", mean_dice)
df = pd.DataFrame(IoU_values, columns=["IoU"])
df = df[df.IoU != 1.0]    
mean_IoU = df.mean().values
print("Mean IoU is: ", mean_IoU)   



## cbis testing samples


model= att_unet_model

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







# patches prediction

# from PIL import Image
# import numpy as np
# from matplotlib import pyplot as plt
# from patchify import patchify, unpatchify
# from tensorflow.keras.utils import normalize
# #Apply a trained model on large image
# large_image = cv2.imread('G:/BreastCancerWorking/Datasets/INbreast/Augmented2_INbreast/All Images/20587612_v_flipped.jpg', 0)
# #This will split the image into small images of shape [3,3]
# patches = patchify(large_image, (256, 256), step=256)  #Step=256 for 256 patches means no overlap

# predicted_patches = []
# for i in range(patches.shape[0]):
#     for j in range(patches.shape[1]):
#         print(i,j)
        
#         single_patch = patches[i,j,:,:]
#         single_patch = cv2.cvtColor(single_patch, cv2.COLOR_GRAY2RGB)
#         single_patch_norm = normalize(np.array(single_patch), axis=1)
#         single_patch_input=np.expand_dims(single_patch_norm, 0)

# #Predict and threshold for values above 0.5 probability

#         single_patch_prediction = (model.predict(single_patch_input)[0,:,:,0] > 0.5).astype(np.uint8)
#         predicted_patches.append(single_patch_prediction)

# predicted_patches = np.array(predicted_patches)

# predicted_patches_reshaped = np.reshape(predicted_patches, (patches.shape[0], patches.shape[1], 256,256) )
# reconstructed_image = unpatchify(predicted_patches_reshaped, large_image.shape)
# plt.imshow(reconstructed_image, cmap='gray')
# plt.imsave('G:/python_for_microscopists-master/Inbreast/INBreastResults/dilatedUnet/segm.jpg', reconstructed_image, cmap='gray')

# plt.hist(reconstructed_image.flatten())  #Threshold everything above 0

# # final_prediction = (reconstructed_image > 0.01).astype(np.uint8)
# # plt.imshow(final_prediction)

# plt.figure(figsize=(8, 8))
# plt.subplot(221)
# plt.title('Large Image')
# plt.imshow(large_image, cmap='gray')
# plt.subplot(222)
# plt.title('Prediction of large Image')
# plt.imshow(reconstructed_image, cmap='gray')
# plt.show()



import pandas as pd    
unet_history_df = pd.DataFrame(unet_history.history) 
att_unet_history_df = pd.DataFrame(AttentionUnet_history.history) 
att_res_unet_history_df = pd.DataFrame(Attention_ResUNet_history.history) 
unetpp_cbis_history_df = pd.DataFrame(unetpp_model_history.history) 
with open('unet2_history_df.csv', mode='w') as f:
    unet_history_df.to_csv(f)
    
with open('att_unet_history_df.csv', mode='w') as f:
    att_unet_history_df.to_csv(f)

with open('custom_code_att_res_unet_history_df.csv', mode='w') as f:
    att_res_unet_history_df.to_csv(f)    

with open('unetpp_cbis_history_df.csv', mode='w') as f:
    unetpp_cbis_history_df.to_csv(f) 
#######################################################################
#Check history plots, one model at a time
history = unet_history
history = AttentionUnet_history
history = Attention_ResUNet_history
history = unetpp_model_history

#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['jacard_coef']
#acc = history.history['accuracy']
val_acc = history.history['val_jacard_coef']
#val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training Jacard')
plt.plot(epochs, val_acc, 'r', label='Validation Jacard')
plt.title('Training and validation Jacard')
plt.xlabel('Epochs')
plt.ylabel('Jacard')
plt.legend()
plt.show()

