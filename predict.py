# Import Libraries
from loss import soft_dice_loss, iou_coef
import matplotlib.pyplot as plt
from skimage.io import imread
import numpy as np


# Load saved model
model = load_model("./Trained_Model/Road_Model.h5", custom_objects={'soft_dice_loss': soft_dice_loss, 'iou_coef': iou_coef})


test_path = 'test_image.jpg'
mask_path = 'test_image.jpg'
test_img  = np.asarray([imread(test_path)])

f = plt.figure(figsize = (24, 20))
f.add_subplot(1,3,1)
plt.imshow(imread(test_path), cmap='gray')
plt.title("Input Image")
plt.axis('off')

f.add_subplot(1,3,2)
plt.imshow(imread(mask_path), cmap='gray')
plt.title("Ground Truth")
plt.axis('off')

f.add_subplot(1,3,3)
plt.imshow(model.predict(test_img, verbose=1)[0][:,:,0], cmap='gray')
plt.title("Predicted Image")
plt.axis('off')

plt.show()