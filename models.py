import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.backend import learning_phase
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from layers.BayesDropout import BayesDropout
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as K
from tensorflow.keras.layers import LeakyReLU

def unet(pretrained_weights = None,input_size = (256,256,1), loss_function = binary_crossentropy):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, padding = 'same', activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = 1e-5), loss = binary_crossentropy, metrics = ['accuracy'])

    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model

def unet_Enze19(pretrained_weights = None,input_size = (256,256,1), loss_function = binary_crossentropy):
    # this model is based on the following paper by Enze Zhang et al.:
    # Automatically delineating the calving front of Jakobshavn Isbræ from multitemporal TerraSAR-X images: a deep learning approach
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'linear', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU(alpha=.001)(conv1)


    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, padding = 'same', activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = 1e-5), loss = binary_crossentropy, metrics = ['accuracy'])

    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model


def unet_Enze19_2(pretrained_weights = None,input_size = (256,256,1), loss_function = binary_crossentropy):
    # this model is based on the following paper by Enze Zhang et al.:
    # Automatically delineating the calving front of Jakobshavn Isbræ from multitemporal TerraSAR-X images: a deep learning approach
    inputs = Input(input_size)
    conv1 = Conv2D(32, 5, padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU(alpha=.1)(conv1)
    conv1 = Conv2D(32, 5, padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU(alpha=.1)(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) # 128

    conv2 = Conv2D(64, 5, padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU(alpha=.1)(conv2)
    conv2 = Conv2D(64, 5, padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU(alpha=.1)(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) # 64

    conv3 = Conv2D(128, 5, padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU(alpha=.1)(conv3)
    conv3 = Conv2D(128, 5, padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU(alpha=.1)(conv3)

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3) # 32

    conv4 = Conv2D(256, 5, padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = LeakyReLU(alpha=.1)(conv4)
    conv4 = Conv2D(256, 5, padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = LeakyReLU(alpha=.1)(conv4)

    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4) # 16

    conv5 = Conv2D(512, 5, padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = LeakyReLU(alpha=.1)(conv5)
    conv5 = Conv2D(512, 5, padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = LeakyReLU(alpha=.1)(conv5)

    up6 = Conv2DTranspose(256, 5,  strides=(2, 2), padding = 'same', kernel_initializer = 'he_normal')(conv5)
    merge6 = concatenate([conv4,up6], axis = 3)

    conv6 = Conv2D(256, 5, padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = LeakyReLU(alpha=.1)(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(256, 5, padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = LeakyReLU(alpha=.1)(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = Conv2DTranspose(128, 5,  strides=(2, 2), padding = 'same', kernel_initializer = 'he_normal')(conv6)
    merge7 = concatenate([conv3,up7], axis = 3)

    conv7 = Conv2D(128, 5, padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = LeakyReLU(alpha=.1)(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(128, 5, padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = LeakyReLU(alpha=.1)(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = Conv2DTranspose(64, 5,  strides=(2, 2), padding = 'same', kernel_initializer = 'he_normal')(conv7)
    merge8 = concatenate([conv2,up8], axis = 3)

    conv8 = Conv2D(64, 5, padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = LeakyReLU(alpha=.1)(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(64, 5, padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = LeakyReLU(alpha=.1)(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = Conv2DTranspose(32, 5,  strides=(2, 2), padding = 'same', kernel_initializer = 'he_normal')(conv8)
    merge9 = concatenate([conv1,up9], axis = 3)

    conv9 = Conv2D(32, 5, padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = LeakyReLU(alpha=.1)(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(32, 5, padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = LeakyReLU(alpha=.1)(conv9)
    conv9 = BatchNormalization()(conv9)

    conv10 = Conv2D(1, 3, padding = 'same', activation = 'sigmoid')(conv9)

    model = Model(inputs = inputs, outputs = conv10, name='unet_Enze19_2')

    model.compile(optimizer = Adam(lr = 1e-4), loss = loss_function, metrics = ['accuracy'])

    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model

def unet_Enze19_2_bayes(pretrained_weights = None,input_size = (256,256,1), output_channels=1, loss_function = binary_crossentropy, drop_rate=0.5):
    # this model is based on the following paper by Enze Zhang et al.:
    # Automatically delineating the calving front of Jakobshavn Isbræ from multitemporal TerraSAR-X images: a deep learning approach
    inputs = Input(input_size)
    conv1 = Conv2D(32, 5, padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU(alpha=.1)(conv1)
    conv1 = Conv2D(32, 5, padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU(alpha=.1)(conv1)
    drop1 = BayesDropout(rate=drop_rate, name='drop1')(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(drop1) # 128

    conv2 = Conv2D(64, 5, padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU(alpha=.1)(conv2)
    conv2 = Conv2D(64, 5, padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU(alpha=.1)(conv2)
    drop2 = BayesDropout(rate=drop_rate, name='drop2')(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(drop2) # 64

    conv3 = Conv2D(128, 5, padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU(alpha=.1)(conv3)
    conv3 = Conv2D(128, 5, padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU(alpha=.1, name='conv3')(conv3)
    drop3 = BayesDropout(rate=drop_rate, name='drop3')(conv3)

    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3) # 32

    conv4 = Conv2D(256, 5, padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = LeakyReLU(alpha=.1)(conv4)
    conv4 = Conv2D(256, 5, padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = LeakyReLU(alpha=.1, name='conv4')(conv4)
    drop4 = BayesDropout(rate=drop_rate, name='drop4')(conv4)

    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4) # 16

    conv5 = Conv2D(512, 5, padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = LeakyReLU(alpha=.1)(conv5)
    conv5 = Conv2D(512, 5, padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = LeakyReLU(alpha=.1)(conv5)

    up6 = Conv2DTranspose(256, 5,  strides=(2, 2), padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop6 = BayesDropout(rate=drop_rate, name='drop6')(up6)
    merge6 = concatenate([drop4,drop6], axis = 3)

    conv6 = Conv2D(256, 5, padding = 'same', kernel_initializer = 'he_normal', name='merge6')(merge6)
    conv6 = LeakyReLU(alpha=.1)(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(256, 5, padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = LeakyReLU(alpha=.1)(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = Conv2DTranspose(128, 5,  strides=(2, 2), padding = 'same', kernel_initializer = 'he_normal')(conv6)
    drop7 = BayesDropout(rate=drop_rate, name='drop7')(up7)
    merge7 = concatenate([drop3,drop7], axis = 3)

    conv7 = Conv2D(128, 5, padding = 'same', kernel_initializer = 'he_normal', name='merge7')(merge7)
    conv7 = LeakyReLU(alpha=.1)(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(128, 5, padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = LeakyReLU(alpha=.1)(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = Conv2DTranspose(64, 5,  strides=(2, 2), padding = 'same', kernel_initializer = 'he_normal')(conv7)
    drop8 = BayesDropout(rate=drop_rate, name='drop8')(up8)
    merge8 = concatenate([drop2,drop8], axis = 3)

    conv8 = Conv2D(64, 5, padding = 'same', kernel_initializer = 'he_normal', name='merge8')(merge8)
    conv8 = LeakyReLU(alpha=.1)(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(64, 5, padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = LeakyReLU(alpha=.1)(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = Conv2DTranspose(32, 5,  strides=(2, 2), padding = 'same', kernel_initializer = 'he_normal')(conv8)
    drop9 = BayesDropout(rate=drop_rate, name='drop9')(up9)
    merge9 = concatenate([drop1,drop9], axis = 3)

    conv9 = Conv2D(32, 5, padding = 'same', kernel_initializer = 'he_normal', name='merge9')(merge9)
    conv9 = LeakyReLU(alpha=.1)(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(32, 5, padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = LeakyReLU(alpha=.1)(conv9)
    conv9 = BatchNormalization()(conv9)

    conv10 = Conv2D(output_channels, 3, padding = 'same', activation = 'sigmoid')(conv9)

    model = Model(inputs = inputs, outputs = conv10, name='unet_Enze19_bayes')

    model.compile(optimizer = Adam(lr = 1e-4), loss = loss_function, metrics = ['accuracy'])

    #model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

