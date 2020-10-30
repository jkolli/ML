# 
# TAU Ethiopic Digit Recognition
# https://www.kaggle.com/c/tau-ethiopic-digit-recognition/overview
# 

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from skimage.transform import rescale, resize, downscale_local_mean
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models


# Import the training data from "data/train"-folder
def import_training_data(path):
    # Generate data generator for image processing, 
    # rescale is used for data normalization
    data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        data_format="channels_last", rescale=1./255, validation_split=0.2)
    
    # Read images from folders to datasets
    # Speficy correct class names, 0 is not used
    class_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    train_ds = data_generator.flow_from_directory(
        path, target_size=(28, 28), color_mode="grayscale", 
        classes=class_names, class_mode="categorical", subset="training")
    train_ds.labels[:] + 1
    valid_ds = data_generator.flow_from_directory(
        path, target_size=(28, 28), color_mode="grayscale", 
        classes=class_names, class_mode="categorical", subset="validation")
    return train_ds, valid_ds


# Import the testing data from "data/test"-folder
def import_testing_data(path):
    data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        data_format="channels_last", rescale=1./255)
    test_ds = data_generator.flow_from_directory(
        path, target_size=(28, 28), color_mode="grayscale", 
        class_mode=None, shuffle=False)
    return test_ds


# Define CNN model
def define_model():
    
    # Define input shape to image shape
    img_inputs = keras.Input(shape=(28, 28, 1))

    # Define layers
    x = layers.Conv2D(64, (3,3), activation="relu",
                                kernel_initializer="he_uniform",
                                padding="same")(img_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3,3), activation="relu",
                            kernel_initializer="he_uniform",
                            padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2,2), padding="valid")(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, (3,3), activation="relu",
                            kernel_initializer="he_uniform",
                            padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3,3), activation="relu",
                            kernel_initializer="he_uniform",
                            padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3,3), activation="relu",
                            kernel_initializer="he_uniform",
                            padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2,2), padding="valid")(x)
    x = layers.Dropout(0.4)(x)

    x = layers.GlobalAveragePooling2D()(x)
    # x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu",
                        kernel_initializer="he_uniform")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu",
                        kernel_initializer="he_uniform")(x)
    x = layers.Dropout(0.5)(x)
    lab_ouputs = layers.Dense(10, activation="softmax")(x)
    
    model = keras.Model(inputs=img_inputs, outputs=lab_ouputs, name="MNIST-model")
    return model


# Define the compile attributes for the model
def compile_model(model):
    model.compile(
        loss = 'categorical_crossentropy',
        optimizer = keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.8),
        metrics=['accuracy'])
    return model


# Fit the CNN model
def fit_model(train_ds, valid_ds, model, ep, bs):
    history = model.fit(
        train_ds,
        batch_size=bs,
        epochs=ep,
        validation_data=valid_ds)
    return history


# Plot the training results
def plot_training_result(history):
    plt.subplot(211)
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.title('Cross Entropy Loss (blue=train, red=test)')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='red', label='test')
    plt.grid(True)
    plt.subplot(212)
    plt.title('Accuracy (blue=train, red=test)')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.plot(history.history['acc'], color='blue', label='train')
    plt.plot(history.history['val_acc'], color='red', label='test')
    plt.grid(True)
    plt.show()


# Create result .csv for Kaggle
def output_result_data(pred_labels, filename):
    # Create file path
    file_path = os.path.join('results/', filename)

    # Output results
    with open(file_path, "w") as output_file: 
        output_file.write("Id,Category\n") 
        for idx in range(10000): 
            output_file.write(f"{idx:05},{pred_labels[idx]}\n")


#
# Main
#
if __name__ == '__main__':
    
    #####################
    # USER PARAMETERS   #
    #####################
    ep = 100  # Number of epochs
    bs = 32  # Batch size

    # Start run
    start_time = time.time()
    print('Starting process {}...'.format(
        time.strftime('%X', time.localtime())))
    print('Tensorflow version: ', tf.__version__)
    print('Keras version: ', keras.__version__)

    # Read data files
    print('Importing image files...')
    train_ds, valid_ds = import_training_data('data/train/train')
    test_ds = import_testing_data('data/test')
    print('Image files ready.')

    # Define model functions  
    model = define_model()
    model.summary()
    model = compile_model(model)
    history = fit_model(train_ds, valid_ds, model, ep, bs)
    
    # Store training results
    train_time = time.time()-start_time
    print('Training ready at {}.'.format(
        time.strftime('%X', time.localtime())))
    print('Run time: {:2.2f} s.'.format(
        train_time))
    # plot_training_result(history)
    
    # Predict labels
    print('Predicting test labels...')
    pred_labs = model.predict(test_ds)
    pred_labs = np.argmax(pred_labs, axis=1)

    # Scale labels 0-9 -> 1-10
    pred_labs = np.add(pred_labs, 1)
    print('Labels ready.')

    # Output result file
    print('Creating result file...')
    output_result_data(pred_labs, 'kaggle_submission.csv')
    print('Results ready.')
