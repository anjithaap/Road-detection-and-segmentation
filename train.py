# Import Libraries
from keras.callbacks import ModelCheckpoint
from IPython.display import clear_output
import matplotlib.pyplot as plt
from skimage.io import imread
import tensorflow as tf
import numpy as np


# Import Prepared Dataset
from prepare_dataset import getDataset

# Import loss functions
from loss import soft_dice_loss

# Import Model architecture
from models.fcn_32 import FCN32
from models.fcn_8 import FCN8

# Get prepared dataset
train_images, train_masks = getDataset()

# Define model network to be trained
model = FCN32()     # Default model set as FCN32
# model = FCN8()    # Uncomment to use FCN8 network


# Path to save model
model_path = "./Trained-Model/Road_Model.h5"

# Function to save best model weights
checkpointer = ModelCheckpoint(model_path, monitor="val_loss", mode="min", save_best_only = True, verbose=1)


# Code block to show predictions at each epoch
def show_predictions(epoch):
    test_path1 = 'Dataset/Test_Images/11740_sat.jpg'
    test_path2 = 'Dataset/Test_Images/112348_sat.jpg'
    test_path3 = 'Dataset/Test_Images/115172_sat.jpg'

    test_img1  = np.asarray([imread(test_path1)])
    test_img2  = np.asarray([imread(test_path2)])
    test_img3  = np.asarray([imread(test_path3)])

    f = plt.figure(figsize = (8, 10))
    f.suptitle(f'Epoch: {epoch}', x=0.5, y=0.02)

    
    f.add_subplot(3,2,1)
    plt.imshow(imread(test_path1), cmap='gray')
    plt.title("Input Image")
    plt.axis('off')
    f.add_subplot(3,2,2)
    plt.imshow(model.predict(test_img1, verbose=1)[0][:,:,0], cmap='gray')
    plt.title("Predicted Image")
    plt.axis('off')

    f.add_subplot(3,2,3)
    plt.imshow(imread(test_path2), cmap='gray')
    plt.axis('off')
    f.add_subplot(3,2,4)
    plt.imshow(model.predict(test_img2, verbose=1)[0][:,:,0], cmap='gray')
    plt.axis('off')

    f.add_subplot(3,2,5)
    plt.imshow(imread(test_path3), cmap='gray')
    plt.axis('off')
    f.add_subplot(3,2,6)
    plt.imshow(model.predict(test_img3, verbose=1)[0][:,:,0], cmap='gray')
    plt.axis('off')
    
    
    plt.savefig(f'epochs/{epoch}.png')
    plt.show()


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions(epoch+1)



# Define model training parameters
EPOCHS = 100
LEARNING_RATE = 0.0001
BATCH_SIZE = 56
adam = tf.keras.optimizers.Adam(LEARNING_RATE)


# Compile model
model.compile(optimizer=adam, loss=soft_dice_loss, metrics=['accuracy'])


# Start model training
history = model.fit(train_images,
                    train_masks/255,
                    validation_split = 0.1,
                    epochs = EPOCHS,
                    batch_size = BATCH_SIZE,
                    callbacks = [checkpointer, DisplayCallback()])




# Plot training history (epoch accuracy and loss)
history_fig = plt.figure(figsize=(20,5))

accuracy = history_fig.add_subplot(1,2,1)
imgplot = plt.plot(history.history['accuracy'])
imgplot = plt.plot(history.history['val_accuracy'])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training Accuracy', 'Validation accuracy'], loc='upper right')
accuracy.set_title("Epoch Accuracy")

loss = history_fig.add_subplot(1,2,2)
imgplot = plt.plot(history.history['loss'])
imgplot = plt.plot(history.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training loss', 'Validation loss'], loc='upper right')
loss.set_title("Epoch Loss")