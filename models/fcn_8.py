# Import Libraries
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, Activation, BatchNormalization, add
from tensorflow.keras.models import Model

def FCN8():

    img_input = Input(shape=(512, 512, 3))

    x = Conv2D(64, 3, activation='relu', name='Block-1_Conv-1', padding='same') (img_input)
    x = BatchNormalization() (x); x = Dropout(0.1) (x)
    x = Conv2D(64, 3, activation='relu', name='Block-1_Conv-2', padding='same') (x)
    x = BatchNormalization() (x); x = Dropout(0.1) (x)
    x = MaxPooling2D(2, strides=2, name='Pooling-1') (x)
    skip1 = x
    skip1 = Conv2D(1, 1, kernel_initializer='he_normal', name='S-1') (skip1)

    # Block 2
    x = Conv2D(128, 3, activation='relu', name='Block-2_Conv-1', padding='same') (x)
    x = BatchNormalization() (x); x = Dropout(0.2) (x)
    x = Conv2D(128, 3, activation='relu', name='Block-2_Conv-2', padding='same') (x)
    x = BatchNormalization() (x); x = Dropout(0.2) (x)
    x = MaxPooling2D(2, strides=2, name='Pooling-2') (x)
    skip2 = x
    skip2 = Conv2D(1, 1, kernel_initializer='he_normal', name='S-2') (skip2)

    # Block 3
    x = Conv2D(256, 3, activation='relu', name='Block-3_Conv-1', padding='same') (x)
    x = BatchNormalization() (x); x = Dropout(0.3) (x)
    x = Conv2D(256, 3, activation='relu', name='Block-3_Conv-2', padding='same') (x)
    x = BatchNormalization() (x); x = Dropout(0.3) (x)
    x = Conv2D(256, 3, activation='relu', name='Block-3_Conv-3', padding='same') (x)
    x = BatchNormalization() (x); x = Dropout(0.3) (x)
    x = MaxPooling2D(2, strides=2, name='Pooling-3') (x)
    skip3 = x
    skip3 = Conv2D(1, 1, kernel_initializer='he_normal', name='S-3') (skip3)
    

    # Block 4
    x = Conv2D(512, 3, activation='relu', name='Block-4_Conv-1', padding='same') (x)
    x = BatchNormalization() (x); x = Dropout(0.3) (x)
    x = Conv2D(512, 3, activation='relu', name='Block-4_Conv-2', padding='same') (x)
    x = BatchNormalization() (x); x = Dropout(0.3) (x)
    x = Conv2D(512, 3, activation='relu', name='Block-4_Conv-3', padding='same') (x)
    x = BatchNormalization() (x); x = Dropout(0.3) (x)
    x = MaxPooling2D(2, strides=2, name='Pooling-4') (x)
    skip4 = x
    skip4 = Conv2D(1, 1, kernel_initializer='he_normal', name='S-4') (skip4)
    

    # Block 5
    x = Conv2D(512, 3, activation='relu', name='Block-5_Conv-1', padding='same') (x)
    x = BatchNormalization() (x); x = Dropout(0.2) (x)
    x = Conv2D(512, 3, activation='relu', name='Block-5_Conv-2', padding='same') (x)
    x = BatchNormalization() (x); x = Dropout(0.2) (x)
    x = Conv2D(512, 3, activation='relu', name='Block-5_Conv-3', padding='same') (x)
    x = BatchNormalization() (x); x = Dropout(0.2) (x)
    x = MaxPooling2D(2, strides=2, name='Pooling-5') (x)


    x = Conv2D(4096 , (7, 7) , activation='relu' , name='Fully-Connected-1', padding='same') (x)
    x = Conv2D(4096 , (1, 1) , activation='relu' , name='Fully-Connected-2', padding='same') (x)

    
    # Skip connections
    x = Conv2DTranspose(512, kernel_size=2, name='Upsample_2x', strides=2) (x)
    skip4 = MaxPooling2D(2, strides=2) (skip3)
    add4 = add([skip4, x])

    x = Conv2DTranspose(256, kernel_size=2, name='Upsample_4x', strides=2) (add4)
    skip3 = MaxPooling2D(2, strides=2) (skip2)
    add3 = add([skip3, x])

    x = Conv2DTranspose(128, kernel_size=2, kernel_initializer='he_normal', name='Upsample_8x', strides=2) (add3)
    x = Conv2DTranspose( 64, kernel_size=2, kernel_initializer='he_normal', name='Upsample_16x', strides=2) (x)
    x = Conv2DTranspose( 32, kernel_size=2, kernel_initializer='he_normal', name='Upsample_32x', strides=2) (x)

    x = Conv2D(1, 1, kernel_initializer='he_normal') (x)
    x = Dropout(0.1) (x)

    x = (Activation('sigmoid'))(x)
    model = Model(img_input, x)
    return model