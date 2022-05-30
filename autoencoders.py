# -*- coding: utf-8 -*-
"""
DESCRIPTION:
This file contains shape-symmetric convolutional autoencoder architectures.
The image shape is (74, 200, 1).

REFERENCES :
    
https://github.com/nlinc1905/Convolutional-Autoencoder-Music-Similarity/blob/master/03_autoencoding_and_tsne.py

https://analyticsindiamag.com/how-to-implement-convolutional-autoencoder-in-pytorch-with-cuda/

https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2DTranspose

https://www.reddit.com/r/MachineLearning/comments/ef1xe8/d_should_autoencoders_really_be_symmetric/
    
https://stackoverflow.com/questions/61614366/convolutional-autoencoders

"""
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Conv2DTranspose, Reshape, Dropout
from keras.models import Model
#from tensorflow import keras

img_length = 74
img_height = 200
img_channels = 1


def autoencoder_1(latent_dim):
    
    #ENCODER
    enc_input = Input((img_length, img_height, img_channels))
    enc = Conv2D(2,  (2, 2), strides=2, activation='relu')(enc_input)
    enc = Dropout(0.2)(enc)
    enc = Conv2D(8,  (4, 4), strides=4, activation='relu')(enc)
    enc = Dropout(0.2)(enc)
    enc = Conv2D(8,  (2, 2), strides=2, activation='relu')(enc)
    enc = Flatten()(enc)
    
    #BOTTLENECK
    code_features = Dense(latent_dim, activation='softmax')(enc)
    encoder = Model(enc_input, code_features)
    #encoded_imgs = encoder.predict(x_train)
    
    #DECODER
    dec_input = code_features
    dec = Dense(384)(dec_input) # to 1280 exei prokupsei exwntas dei th diastash tou flatten vector ston encoder apopanw
    dec = Reshape((4, 12, 8))(dec)
    dec = Conv2DTranspose(8, (2, 2), strides=2, activation='relu', output_padding = (1,1))(dec)
    dec = Dropout(0.2)(dec)
    dec = Conv2DTranspose(2, (4, 4), strides=4, activation='relu', output_padding = (1,0))(dec)
    dec = Dropout(0.2)(dec)
    dec_output = Conv2DTranspose(img_channels, (2, 2), strides=2, activation='relu')(dec)
    #decoder = Model(code_features, dec_output)
    
    #AUTOENCODER
    autoencoder = Model(enc_input, dec_output)
    
    return autoencoder, encoder

def autoencoder_2(latent_dim):
    
    #ENCODER
    enc_input = Input((img_length, img_height, img_channels))
    enc = Conv2D(2,  (2, 2), strides=2, activation='relu')(enc_input)
    enc = Dropout(0.2)(enc)
    enc = Conv2D(4,  (3, 3), strides=2, activation='relu')(enc)
    enc = Dropout(0.2)(enc)
    enc = Conv2D(8,  (2, 2), strides=2, activation='relu')(enc)
    enc = Dropout(0.2)(enc)
    enc = Conv2D(16,  (2, 2), strides=2, activation='relu')(enc)
    enc = Flatten()(enc)
    
    #BOTTLENECK
    code_features = Dense(latent_dim, activation='softmax')(enc)
    encoder = Model(enc_input, code_features)
    
    #DECODER
    dec_input = code_features
    dec = Dense(768)(dec_input)
    dec = Reshape((4, 12, 16))(dec)
    dec = Conv2DTranspose(8, (2, 2), strides=2, activation='relu', output_padding = (1,0))(dec)
    dec = Dropout(0.2)(dec)
    dec = Conv2DTranspose(4, (2, 2), strides=2, activation='relu', output_padding = (0,1))(dec)
    dec = Dropout(0.2)(dec)
    dec = Conv2DTranspose(2, (3, 3), strides=2, activation='relu', output_padding = (0,1))(dec)
    dec = Dropout(0.2)(dec)
    dec_output = Conv2DTranspose(img_channels, (2, 2), strides=2, activation='relu')(dec)
    #decoder = Model(code_features, dec_output)
    
    #AUTOENCODER
    autoencoder = Model(enc_input, dec_output)
    
    return autoencoder, encoder

def autoencoder_3(latent_dim):
    
    # flatten 1280
    # before/after code_features dense 256 (dif from autoencoder_3)
    
    #ENCODER
    enc_input = Input((img_length, img_height, img_channels))
    enc = Conv2D(16,  (2, 2), strides=2, activation='relu')(enc_input)
    enc = Dropout(0.2)(enc)
    enc = Conv2D(32,  (2, 2), strides=2, activation='relu')(enc)
    enc = Dropout(0.2)(enc)
    enc = Conv2D(64,  (3, 3), strides=3, activation='relu')(enc)
    enc = Dropout(0.2)(enc)
    enc = Conv2D(128, (3, 3), strides=3, activation='relu')(enc)
    
    enc = Flatten()(enc)
    enc = Dense(256, activation = 'relu')(enc)
    
    #BOTTLENECK
    code_features = Dense(latent_dim, activation='softmax')(enc)
    encoder = Model(enc_input, code_features)
    #encoded_imgs = encoder.predict(x_train)
    
    #DECODER
    #dec_input = code_features
    dec = Dense(256, activation='relu')(code_features)
    dec = Dense(1280)(dec) # to 1280 exei prokupsei exwntas dei th diastash tou flatten vector ston encoder apopanw
    dec = Reshape((2, 5, 128))(dec)
    dec = Conv2DTranspose(64, (3, 3), strides=3, activation='relu', output_padding = (0,1))(dec)
    dec = Dropout(0.2)(dec)
    dec = Conv2DTranspose(32, (3, 3), strides=3, activation='relu', output_padding = (0,2))(dec)
    dec = Dropout(0.2)(dec)
    dec = Conv2DTranspose(16, (2, 2), strides=2, activation='relu', output_padding = (1,0))(dec)
    dec = Dropout(0.2)(dec)
    dec_output = Conv2DTranspose(img_channels, (2, 2), strides=2, activation='relu')(dec)
    #decoder = Model(code_features, dec_output)
    
    #AUTOENCODER
    autoencoder = Model(enc_input, dec_output)
    
    return autoencoder, encoder

def autoencoder_4(latent_dim):
    
    # flatten 3072
    # before/after code_features dense 256
    
    #ENCODER
    enc_input = Input((img_length, img_height, img_channels))
    enc = Conv2D(8,  (4, 4), strides=2, activation='relu')(enc_input)
    enc = Dropout(0.2)(enc)
    enc = Conv2D(16,  (3, 3), strides=2, activation='relu')(enc)
    enc = Dropout(0.2)(enc)
    enc = Conv2D(32, (2, 2), strides=2, activation='relu')(enc)
    enc = Dropout(0.2)(enc)
    enc = Conv2D(64, (2, 2), strides=2, activation='relu')(enc)
    
    enc = Flatten()(enc)
    enc = Dense(256, activation = 'relu')(enc)
    code_features = Dense(latent_dim, activation='softmax')(enc)
    
    encoder = Model(enc_input, code_features)
    #encoded_imgs = encoder.predict(x_train)
    
    #DECODER
    #dec_input = code_features
    dec = Dense(256, activation='relu')(code_features)
    dec = Dense(3072)(dec)
    dec = Reshape((4, 12, 64))(dec)
    dec = Conv2DTranspose(32, (2, 2), strides=2, activation='relu')(dec)
    dec = Dropout(0.2)(dec)
    dec = Conv2DTranspose(16, (2, 2), strides=2, activation='relu', output_padding = (1,1))(dec)
    dec = Dropout(0.2)(dec)
    dec = Conv2DTranspose(8, (3, 3), strides=2, activation='relu', output_padding = (1,0))(dec)
    dec = Dropout(0.2)(dec)
    dec_output = Conv2DTranspose(img_channels, (4, 4), strides=2, activation='relu')(dec)
    #decoder = Model(code_features, dec_output)
    
    #AUTOENCODER
    autoencoder = Model(enc_input, dec_output)
    
    return autoencoder, encoder