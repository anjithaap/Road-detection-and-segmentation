# Import Libraries
import os
import cv2
import numpy as np
from skimage.io import imread
from sklearn.model_selection import train_test_split

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

images = np.stack(images)
masks = np.stack(masks) / 255

train_images, test_images, train_masks, test_masks = train_test_split(images, masks, test_size=0.2, random_state=2)

pred_images = sample(os.listdir(IMAGES_PATH), 4)
pred_masks  = []
for mask_name in pred_images:
    mask_name = mask_name.replace('_sat.jpg', '_mask.png')
    pred_masks.append(mask_name)

del images, masks

print("Training Set")
print(train_images.shape)
print(train_masks.shape)
print("Testing Set")
print(test_images.shape)
print(test_masks.shape)

def getDataset():
    return (train_images, test_images, train_masks, test_masks, pred_images, pred_masks)