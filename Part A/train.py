import numpy as np
import os
import matplotlib.pyplot as plt
import keras
from keras.layers import Conv2D , MaxPool2D , Flatten , Dropout, Dense, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report,confusion_matrix
import seaborn as sns
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.models import Sequential, load_model
import random
import wandb
import shutil
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from wandb.keras import WandbCallback
from keras.utils.vis_utils import plot_model

#Set up the training, validation and test generators
def generators(train_batch_size, data_aug):
    if data_aug:
        train_datagen = ImageDataGenerator(rescale=1./255,
                                        height_shift_range=0.2,
                                        width_shift_range=0.2,
                                        horizontal_flip=True,
                                        zoom_range=0.2,
                                        fill_mode="nearest")
    else:
        train_datagen = ImageDataGenerator(rescale=1./255)

    val_datagen = ImageDataGenerator(rescale=1./255)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        'inaturalist_12K/train',
        target_size=input_image_shape[:2],
        color_mode="rgb",
        batch_size=train_batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=42)

    # batch_size for validation and test generator should perfectly divide the total number of examples
    validation_generator = val_datagen.flow_from_directory(
        'inaturalist_12K/validation',
        target_size=input_image_shape[:2],
        color_mode="rgb",
        batch_size=100,
        class_mode='categorical',
        shuffle=True,
        seed=42)

    test_generator = test_datagen.flow_from_directory(
        'inaturalist_12K/val',
        target_size=input_image_shape[:2],
        color_mode="rgb",
        batch_size=100,
        class_mode=None,
        shuffle=False,
        seed=42)
    
    return train_generator, validation_generator, test_generator

input_image_shape = (224, 224, 3)

#Building the model
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


def train_validate_model(train_batch_size, data_aug, activation_function_conv, activation_function_dense, num_filters, shape_of_filters_conv, shape_of_filters_pool, batch_norm_use, fc_layer, dropout):
    
    # Create the data generators
    if data_aug:
        train_datagen = ImageDataGenerator(rescale=1./255,
                                        height_shift_range=0.2,
                                        width_shift_range=0.2,
                                        horizontal_flip=True,
                                        zoom_range=0.2,
                                        fill_mode="nearest")
    else:
        train_datagen = ImageDataGenerator(rescale=1./255)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        'inaturalist_12K/train',
        target_size=input_image_shape[:2],
        color_mode="rgb",
        batch_size=train_batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=42)

    test_generator = test_datagen.flow_from_directory(
        'inaturalist_12K/val',
        target_size=input_image_shape[:2],
        color_mode="rgb",
        batch_size=train_batch_size,
        class_mode=None,
        shuffle=False,
        seed=42)
    
    # Define the model
    model = define_model(activation_function_conv, activation_function_dense, num_filters, shape_of_filters_conv, shape_of_filters_pool, batch_norm_use, fc_layer, dropout)

    # Compute the validation step size
    TRAIN_STEP_SIZE = train_generator.n//train_generator.batch_size

    model.compile(optimizer=Adam(lr=0.0001), loss = 'categorical_crossentropy', metrics = ['accuracy'])

    history = model.fit(train_generator,
                        steps_per_epoch = TRAIN_STEP_SIZE,
                        epochs=10,
                        verbose=2)
    
    return history, model


def train_validate_model_wandb():
    # Default values for hyper-parameters
    config_defaults = {
        "data_aug": True,
        "train_batch_size": 128,
        "batch_norm_use": True,
        "dropout": 0,
        "num_filters": [16, 32, 64, 128, 256],
        "fc_layer": 256,
        "shape_of_filters_conv": [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]
    }

    # Initialize a new wandb run
    wandb.init(config=config_defaults)
    
    # Config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config

    # Local variables, values obtained from wandb config
    num_filters = config.num_filters
    data_aug = config.data_aug
    train_batch_size = config.train_batch_size
    batch_norm_use = config.batch_norm_use
    dropout = config.dropout
    fc_layer = config.fc_layer
    shape_of_filters_conv = config.shape_of_filters_conv
    
    # Display the hyperparameters
    run_name = "aug_{}_bs_{}_bn_{}_drop_{}_fc_{}_fil_{}_shape_{}".format(data_aug, train_batch_size, batch_norm_use, dropout, fc_layer, num_filters, shape_of_filters_conv)
    print(run_name)

    # Create the data generators
    train_generator, validation_generator, test_generator = generators(train_batch_size, data_aug)
    
    # Define the model
    model = define_model(activation_function_conv, activation_function_dense, num_filters, shape_of_filters_conv, shape_of_filters_pool, batch_norm_use, fc_layer, dropout)
    print(model.count_params())

    TRAIN_STEP_SIZE = train_generator.n//train_generator.batch_size
    VALIDATION_STEP_SIZE = validation_generator.n//validation_generator.batch_size

    model.compile(optimizer=Adam(lr=0.0001), loss = 'categorical_crossentropy', metrics = ['accuracy'])

    # Early Stopping callback
    earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min')

    # To save the model with best validation accuracy
    mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=0, save_best_only=True)

    history = model.fit(train_generator,
                        steps_per_epoch = TRAIN_STEP_SIZE,
                        validation_data = validation_generator,
                        validation_steps = VALIDATION_STEP_SIZE,
                        epochs=10, 
                        callbacks=[WandbCallback(data_type="image", generator=validation_generator), earlyStopping, mc],
                        verbose=2)
    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    
    # Meaningful name for the run
    wandb.run.name = run_name
    wandb.run.save()
    wandb.run.finish()
    return history


#Hyperparameter Search using WandB
# These are the hyperparameters that we do not sweep over
activation_function_conv = ["relu", "relu", "relu", "relu", "relu"]
activation_function_dense = "relu"
shape_of_filters_pool = [(2, 2), (2, 2), (2, 2), (2, 2), (2, 2)]

# Sweep configuration
sweep_config = {
  "name": "Sweep 2 CS6910 Assignment 2 - Part A",
  "metric": {
      "name":"val_accuracy",
      "goal": "maximize"
  },
  "method": "bayes",
  "parameters": {
        "data_aug": {
            "values": [True, False]
        },
        "train_batch_size": {
            "values": [128, 256]
        },
        "batch_norm_use": {
            "values": [True, False]
        },
        "dropout": {
            "values": [0.1, 0, 0.2]
        },
        "num_filters": {
            "values": [[16, 32, 64, 128, 256], [32, 64, 128, 256, 512], [32, 32, 32, 32, 32],
                       [256, 128, 64, 32, 16], [64, 128, 256, 512, 1024], [128, 64, 32, 16, 8]]
        },
        "fc_layer": {
            "values": [512,256,128]
        },
        "shape_of_filters_conv": {
            "values": [[(3, 3), (3, 3), (3, 3), (5, 5), (7, 7)],
                       [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)],
                       [(7, 7), (5, 5), (3, 3), (3, 3), (3, 3)]]
        }
    }
}

# Generates a sweep id
sweep_id = wandb.sweep(sweep_config, project="Assg-2", entity="kunal_patil")
wandb.agent(sweep_id, train_validate_model_wandb, count=100)
