import numpy as np
# np.random.seed(42)
import tensorflow as tf
# tf.random.set_seed(42)
from tensorflow import keras
# keras.utils.set_random_seed(42)
from keras import models, layers
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, SeparableConv2D
from keras.regularizers import l2
from pathlib import Path



def sb_cnn(input_shape, output_shape):
    """
    Implements SB-CNN model from
    Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification
    Salamon and Bello, 2016.
    https://arxiv.org/pdf/1608.04363.pdf
    Based on Jon Nordbys implementation:
    https://github.com/jonnor/ESC-CNN-microcontroller/blob/572c319c7ad4d0a98bf210d59b26f6df923c8e7b/microesc/models/sbcnn.py
    """
    frames = 128
    bands = 128
    channels = 1
    n_classes = 32
    conv_size = (5, 5)
    conv_block = 'conv'
    downsample_size = (4, 2)
    fully_connected = 64
    n_stages = None
    n_blocks_per_stage = None
    filters = 24
    kernels_growth = 2
    dropout = 0.5
    use_strides = False

    Conv2 = SeparableConv2D if conv_block == 'depthwise_separable' else Convolution2D
    assert conv_block in ('conv', 'depthwise_separable')
    kernel = conv_size
    if use_strides:
        strides = downsample_size
        pool = (1, 1)
    else:
        strides = (1, 1)
        pool = downsample_size

    block1 = [
        Convolution2D(filters, kernel, padding='same', strides=strides,
                      input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(pool_size=pool),
        Activation('relu'),
    ]
    block2 = [
        Conv2(filters * kernels_growth, kernel, padding='same', strides=strides),
        BatchNormalization(),
        MaxPooling2D(pool_size=pool),
        Activation('relu'),
    ]
    block3 = [
        Conv2(filters * kernels_growth, kernel, padding='valid', strides=strides),
        BatchNormalization(),
        Activation('relu'),
    ]
    backend = [
        Flatten(),

        Dropout(dropout),
        Dense(fully_connected, kernel_regularizer=l2(0.001)),
        Activation('relu'),

        Dropout(dropout),
        Dense(output_shape, kernel_regularizer=l2(0.001)),
        Activation('softmax'),
    ]
    layers = block1 + block2 + block3 + backend
    model = Sequential(layers)

    return model


def vgg19(input_shape, output_shape):
    path = os.path.join(os.path.dirname(__file__), "saved_models", "vgg19")
    if os.path.exists(path):
        vgg = keras.models.load_model(path)
    else:
        vgg = keras.applications.VGG19()
        vgg.save(path)

    model = keras.Sequential()
    for layer in vgg.layers[:-1]:
        model.add(layer)
    for layer in model.layers[:-1]:
        layer.trainable = False
    model.add(keras.layers.Dense(units=output_shape, activation="softmax"))
    return model


def convnext(input_shape, output_shape):
    path = os.path.join(os.path.dirname(__file__), "saved_models", "convnextlarge")
    if os.path.exists(path):
        convnext = keras.models.load_model(path)
    else:
        convnext = keras.applications.convnext.ConvNeXtXLarge()
        convnext.save(path)

    x = convnext.layers[-2].output
    output = keras.layers.Dense(output_shape, activation="softmax")(x)
    model = keras.Model(inputs=convnext.input, outputs=output)

    for layer in model.layers[:-3]:
        layer.trainable = False

    return model

