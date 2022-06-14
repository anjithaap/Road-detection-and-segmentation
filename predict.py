#!/usr/bin/python3


# Import Libraries
from loss_functions import soft_dice_loss, iou_coef
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from skimage.io import imread
import numpy as np


# Load saved model
model = load_model("./Trained_Model/Road_Model.h5", custom_objects={'soft_dice_loss': soft_dice_loss, 'iou_coef': iou_coef}) # Select random image from folder


import os
from random import sample

input = imread(f'Dataset/Images/{sample(os.listdir("Dataset/Images"),1)[0]}')
test_data = np.asarray([input])
output = model.predict(test_data, verbose=0)[0][:,:,0]

for xi in range(len(output)):
  for yi in range(len(output[xi])):
    if output[xi][yi] > 0.1:
      input[xi][yi] = [255, 255, 0]

plt.rcParams["figure.figsize"] = (10,10)
plt.imshow(input)
plt.axis('off')
plt.show()