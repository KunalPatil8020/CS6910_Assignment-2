import numpy as np
import tensorflow as tf
import keras
from keras.layers import Conv2D , MaxPool2D , Flatten , Dropout, Dense, Activation, BatchNormalization
from keras.models import Sequential

input_image_shape = (224, 224, 3)

def define_model(activation_function_conv, activation_function_dense, num_filters, shape_of_filters_conv, shape_of_filters_pool, batch_norm_use, fc_layer, dropout):
    model = Sequential()
    model.add(Conv2D(num_filters[0], shape_of_filters_conv[0], input_shape=input_image_shape))
    if batch_norm_use:
        model.add(BatchNormalization())
    model.add(Activation(activation_function_conv[0]))
    model.add(MaxPool2D(pool_size=shape_of_filters_pool[0], strides = (2, 2)))

# loop for 5 layers
    for i in range(1, 5):
        model.add(Conv2D(num_filters[i], shape_of_filters_conv[i]))
        if batch_norm_use:
            model.add(BatchNormalization())
        model.add(Activation(activation_function_conv[i]))
        model.add(MaxPool2D(pool_size=shape_of_filters_pool[i], strides = (2, 2)))

    model.add(Flatten()) # The flatten layer is essential to convert the feature map into a column vector
    model.add(Dense(fc_layer, activation=activation_function_dense))
    model.add(Dropout(dropout)) # For regularization
    model.add(Dense(10, activation="softmax")) #Activation function
    return model
