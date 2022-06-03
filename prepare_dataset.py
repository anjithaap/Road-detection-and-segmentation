# Import Libraries
import os
import cv2
import numpy as np
from skimage.io import imread

# Define path to dataset sub-folders
IMAGES_PATH = 'Dataset/Images/'
MASKS_PATH  = 'Dataset/Masks/'
TEST_PATH   = 'Dataset/Test_Images/'

# Number of images to use (Larger the number, more RAM required)
N_IMAGES = 1500


sat_imgs = os.listdir(IMAGES_PATH)
msk_imgs = os.listdir(MASKS_PATH)
sat_imgs.sort(), msk_imgs.sort()

images = []
for image in (sat_imgs[:N_IMAGES]):
    data = imread(IMAGES_PATH + image)
    images.append(data)

masks = []
for mask in (msk_imgs[:N_IMAGES]):
    data = imread(MASKS_PATH + mask)
    data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY) # Convert mask images from RGB to Grayscale
    data = np.expand_dims(data, axis=-1)
    masks.append(data)

train_images = np.stack(images)
train_masks = np.stack(masks)


print("Training Set")
print(train_images.shape)
print(train_masks.shape)

def getDataset():
    return (train_images, train_masks)