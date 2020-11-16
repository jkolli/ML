#
# Convolutional neural network for the CIFAR-10 image classification
# https://www.cs.toronto.edu/~kriz/cifar.html
#

import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow import keras
from keras.datasets import cifar10


# Normalize the color values between 0-1
def normalize_values(x_data):
    x_data = np.true_divide(x_data, 255, dtype='float')
    print('x_data normalized.')
    return x_data


# Vectorize 32x32x3 images to 1x3072 form
def vectorize_values(x_data):
    data_size = x_data.shape[0]
    x_data = x_data.reshape(data_size, 1024, 3)
    x_data = np.hstack((x_data[:, :, 0], x_data[:, :, 1], x_data[:, :, 2]))
    return x_data


# Encode class labels to onehot labels
def lab_encoder(labels):
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_labels = onehot_encoder.fit_transform(labels)
    return onehot_labels


# Define CNN model
def define_model(vectorize_data):
    # Create model    
    model = keras.models.Sequential(name='CNN model')

    # Select input shape base on selection
    if vectorize_data:
        model.add(keras.Input(shape = 3072))
    else:
        model.add(keras.Input(shape = (32, 32, 3)))
    
    # Add layers
    model.add(keras.layers.Conv2D(64, (3,3), activation="relu",
                            kernel_initializer="he_uniform",
                            padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(64, (3,3), activation="relu",
                            kernel_initializer="he_uniform",
                            padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2), padding="valid"))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Conv2D(128, (3,3), activation="relu",
                            kernel_initializer="he_uniform",
                            padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(128, (3,3), activation="relu",
                            kernel_initializer="he_uniform",
                            padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2), padding="valid"))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Conv2D(256, (3,3), activation="relu",
                            kernel_initializer="he_uniform",
                            padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(256, (3,3), activation="relu",
                            kernel_initializer="he_uniform",
                            padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2), padding="valid"))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation="relu",
                            kernel_initializer="he_uniform"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(10, activation="softmax"))
    return model


# Define the compile attributes for the model
def compile_model(model):
    model.compile(
        loss = 'categorical_crossentropy',
        optimizer = keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.8),
        metrics=['accuracy'])
    return model


# Fit the CNN model
def fit_model(model, ep, bs, x_test, y_test_hot):
    history = model.fit(
        x_train, y_train_hot,
        batch_size=bs,
        epochs=ep,
        validation_data = (x_test, y_test_hot))
    return history


# Plot test result
def plot_result(history):
    # Plot data
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


#
# Main
#
if __name__ == '__main__':
    
    # User parameters
    normalize_data = True
    vectorize_data = False
    ep = 200  # Number of epochs
    bs = 64  # Batch size

    # Start run
    start_time = time.time()
    print('Starting Neural Network process {}...'.format(
        time.strftime('%X', time.localtime())))
    print('Tensorflow version: ', tf.__version__)
    print('Keras version: ', keras.__version__)

    # Input data batches
    print('Importing image batches...')
    tf.keras.backend.set_image_data_format('channels_last')
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print('Batches imported.')

    # Normalize data
    if (normalize_data):
        x_train = normalize_values(x_train)
        x_test = normalize_values(x_test)

    # Vectorize data
    if (vectorize_data):
        x_train = vectorize_values(x_train)
        x_test = vectorize_values(x_test)

    # Encode label data
    y_train_hot = lab_encoder(y_train)
    y_test_hot = lab_encoder(y_test)

    # Model functions   
    model = define_model(vectorize_data)
    model.summary()
    model = compile_model(model)
    history = fit_model(model, ep, bs, x_test, y_test_hot)
    score = model.evaluate(x_test, y_test_hot, verbose=1)
    
    # Print results
    print('Test loss: ', score[0])
    print('Test accuracy: ', score[1])
    test_time = time.time()-start_time
    print('Neural network testing ready at {}.'.format(
        time.strftime('%X', time.localtime())))
    print('Run time: {:2.2f} s.'.format(
        test_time))
    plot_result(history)
