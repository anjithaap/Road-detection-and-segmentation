# Import Libraries
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, Activation, BatchNormalization
from tensorflow.keras.models import Model

def FCN32():

    img_input = Input(shape=(512, 512, 3))

    x = Conv2D(64, 3, activation='relu', name='Block-1_Conv-1', padding='same') (img_input)
    x = BatchNormalization() (x); x = Dropout(0.1) (x)
    x = Conv2D(64, 3, activation='relu', name='Block-1_Conv-2', padding='same') (x)
    x = BatchNormalization() (x); x = Dropout(0.1) (x)
    x = MaxPooling2D(2, strides=2, name='P-1') (x)

    # Block 2
    x = Conv2D(128, 3, activation='relu', name='Block-2_Conv-1', padding='same') (x)
    x = BatchNormalization() (x); x = Dropout(0.2) (x)
    x = Conv2D(128, 3, activation='relu', name='Block-2_Conv-2', padding='same') (x)
    x = BatchNormalization() (x); x = Dropout(0.2) (x)
    x = MaxPooling2D(2, strides=2, name='P-2') (x)

    # Block 3
    x = Conv2D(256, 3, activation='relu', name='Block-3_Conv-1', padding='same') (x)
    x = BatchNormalization() (x); x = Dropout(0.3) (x)
    x = Conv2D(256, 3, activation='relu', name='Block-3_Conv-2', padding='same') (x)
    x = BatchNormalization() (x); x = Dropout(0.3) (x)
    x = Conv2D(256, 3, activation='relu', name='Block-3_Conv-3', padding='same') (x)
    x = BatchNormalization() (x); x = Dropout(0.3) (x)
    x = MaxPooling2D(2, strides=2, name='P-3') (x)
    

    # Block 4
    x = Conv2D(512, 3, activation='relu', name='Block-4_Conv-1', padding='same') (x)
    x = BatchNormalization() (x); x = Dropout(0.3) (x)
    x = Conv2D(512, 3, activation='relu', name='Block-4_Conv-2', padding='same') (x)
    x = BatchNormalization() (x); x = Dropout(0.3) (x)
    x = Conv2D(512, 3, activation='relu', name='Block-4_Conv-3', padding='same') (x)
    x = BatchNormalization() (x); x = Dropout(0.3) (x)
    x = MaxPooling2D(2, strides=2, name='P-4') (x)
    

    # Block 5
    x = Conv2D(512, 3, activation='relu', name='Block-5_Conv-1', padding='same') (x)
    x = BatchNormalization() (x); x = Dropout(0.2) (x)
    x = Conv2D(512, 3, activation='relu', name='Block-5_Conv-2', padding='same') (x)
    x = BatchNormalization() (x); x = Dropout(0.2) (x)
    x = Conv2D(512, 3, activation='relu', name='Block-5_Conv-3', padding='same') (x)
    x = BatchNormalization() (x); x = Dropout(0.2) (x)
    x = MaxPooling2D(2, strides=2, name='P-5') (x)


    x = Conv2D(4096 , (7, 7) , activation='relu' , name='Fully-Connected-1', padding='same') (x)
    x = Conv2D(4096 , (1, 1) , activation='relu' , name='Fully-Connected-2', padding='same') (x)

    
    # Upsampling Layers

    x = Conv2DTranspose(128, kernel_size=2, kernel_initializer='he_normal', name='Upsample_2x', strides=2) (x)
    x = Conv2DTranspose( 64, kernel_size=2, kernel_initializer='he_normal', name='Upsample_4x', strides=2) (x)
    x = Conv2DTranspose(128, kernel_size=2, kernel_initializer='he_normal', name='Upsample_8x', strides=2) (x)
    x = Conv2DTranspose( 64, kernel_size=2, kernel_initializer='he_normal', name='Upsample_16x', strides=2) (x)
    x = Conv2DTranspose( 32, kernel_size=2, kernel_initializer='he_normal', name='Upsample_32x', strides=2) (x)

    x = Conv2D(1, 1, kernel_initializer='he_normal') (x)
    x = Dropout(0.1) (x)

    x = (Activation('sigmoid'))(x)
    model = Model(img_input, x)
    return model