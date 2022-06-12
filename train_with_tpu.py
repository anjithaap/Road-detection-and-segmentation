#!/usr/bin/python3


# Import Libraries
from keras.callbacks import ModelCheckpoint
from IPython.display import clear_output
import matplotlib.pyplot as plt
from skimage.io import imread
import tensorflow as tf
import numpy as np
import os


# Import Prepared Dataset
from prepare_dataset import getDataset

# Import loss functions
from loss_functions import soft_dice_loss

# Initiate TPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

resolver = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.experimental.TPUStrategy(resolver)

# Import Model architecture
from models.fcn_32 import FCN32
from models.fcn_8 import FCN8

# Get prepared dataset
train_images, test_images, train_masks, test_masks, pred_images, pred_masks = getDataset()

# Define model network to be trained
with strategy.scope():
    model = FCN32()     # Default model set as FCN32
    # model = FCN8()    # Uncomment to use FCN8 network


# Path to save model
model_path = "./Trained-Model/Road_Model.h5"

# Function to save best model weights
checkpointer = ModelCheckpoint(model_path, monitor="val_loss", mode="min", save_best_only = True, verbose=1)


# Code block to show predictions at each epoch
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import backend as K
from IPython.display import clear_output
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf

model_path = "./Trained_Model/Road_Model.h5"


checkpointer = ModelCheckpoint(model_path, monitor="val_loss", mode="min", save_best_only = True, verbose=1)
earlystopper = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 10, verbose = 1, restore_best_weights = True)
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, min_delta=1e-4)

def model_status(epoch, accu, val_accu, loss, val_loss):

    pred_img_path1 = IMAGES_PATH + pred_images[0]; pred_msk_path1 = MASKS_PATH + pred_masks[0]
    pred_img_path2 = IMAGES_PATH + pred_images[1]; pred_msk_path2 = MASKS_PATH + pred_masks[1]
    pred_img_path3 = IMAGES_PATH + pred_images[2]; pred_msk_path3 = MASKS_PATH + pred_masks[2]
    pred_img_path4 = IMAGES_PATH + pred_images[3]; pred_msk_path4 = MASKS_PATH + pred_masks[3]

    test_img1  = np.asarray([imread(pred_img_path1)])
    test_img2  = np.asarray([imread(pred_img_path2)])
    test_img3  = np.asarray([imread(pred_img_path3)])
    test_img4  = np.asarray([imread(pred_img_path4)])

    f = plt.figure(figsize = (24, 16))
    gs = f.add_gridspec(5, 6)
    f.suptitle(f'Epoch: {epoch}', x=0.5, y=0.02)

    
    f.add_subplot(gs[0, 0])
    plt.imshow(imread(pred_img_path1), cmap='gray')
    plt.title("Input Image")
    plt.axis('off')
    f.add_subplot(gs[0, 1])
    plt.imshow(imread(pred_msk_path1), cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    f.add_subplot(gs[0, 2])
    plt.imshow(model.predict(test_img1, verbose=1)[0][:,:,0], cmap='gray')
    plt.title("Predicted Image")
    plt.axis('off')


    f.add_subplot(gs[1, 0])
    plt.imshow(imread(pred_img_path2), cmap='gray')
    plt.axis('off')
    f.add_subplot(gs[1, 1])
    plt.imshow(imread(pred_msk_path2), cmap='gray')
    plt.axis('off')
    f.add_subplot(gs[1, 2])
    plt.imshow(model.predict(test_img2, verbose=1)[0][:,:,0], cmap='gray')
    plt.axis('off')

    f.add_subplot(gs[2, 0])
    plt.imshow(imread(pred_img_path3), cmap='gray')
    plt.axis('off')
    f.add_subplot(gs[2, 1])
    plt.imshow(imread(pred_msk_path3), cmap='gray')
    plt.axis('off')
    f.add_subplot(gs[2, 2])
    plt.imshow(model.predict(test_img3, verbose=1)[0][:,:,0], cmap='gray')
    plt.axis('off')

    f.add_subplot(gs[3, 0])
    plt.imshow(imread(pred_img_path4), cmap='gray')
    plt.axis('off')
    f.add_subplot(gs[3, 1])
    plt.imshow(imread(pred_msk_path4), cmap='gray')
    plt.axis('off')
    f.add_subplot(gs[3, 2])
    plt.imshow(model.predict(test_img4, verbose=1)[0][:,:,0], cmap='gray')
    plt.axis('off')

    f.add_subplot(gs[0:2, 3:6])
    plt.plot(accu)
    plt.plot(val_accu)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.title("Model Accuracy")
    plt.legend(['Training Accuracy', 'Validation accuracy'], loc='lower right')
    
    f.add_subplot(gs[3:5, 3:6])
    plt.plot(loss)
    plt.plot(val_loss)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title("Model Loss")
    plt.legend(['Training loss', 'Validation loss'], loc='upper right')
    
    plt.savefig(f'epochs/{epoch}.png')
    plt.show()



class DisplayCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.accu = []
        self.val_accu = []
        self.loss = []
        self.val_loss = []
    def on_epoch_end(self, epoch, logs=None):
        self.accu.append(logs.get("accuracy"))
        self.val_accu.append(logs.get("val_accuracy"))
        self.loss.append(logs.get("loss"))
        self.val_loss.append(logs.get("val_loss"))
        clear_output(wait=True)
        model_status(epoch, self.accu, self.val_accu, self.loss, self.val_loss)



# Define model training parameters
EPOCHS = 100
LEARNING_RATE = 0.0001
BATCH_SIZE = 56
adam = tf.keras.optimizers.Adam(LEARNING_RATE)


# Compile model
model.compile(optimizer=adam, loss=soft_dice_loss, metrics=['accuracy'])


# Start model training
history = model.fit(train_images,
                    train_masks,
                    validation_data = (test_images, test_masks),
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