import os
import numpy as np

from keras import models
from keras import layers

from keras.datasets import mnist
from keras.utils import to_categorical

# Use to grab a previously saved model (.h5 file)
from keras.models import load_model

# Main training logic goes here.
def train_and_save(run):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Simple dense model
    model = models.Sequential()
    model.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["accuracy"])

    # Flatten image data, current shape is (nsamples, 28, 28), we want (nsamples, 28*28,)
    X_train = X_train.reshape((len(X_train), 28*28))
    X_test = X_test.reshape((len(X_test), 28*28))

    # Then we normalize our channel data.
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    # One-hot encoding training and testing labels
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Access history of loss and accuracy across all epochs
    hist = model.fit(X_train, y_train, epochs=5, batch_size=128)

    acc = model.evaluate(X_test, y_test)
    acc = str(acc[1])[2:] # Accuracy of model when training is finished, needed?
    
    model_name = "keras-mnist-{}".format(acc)
    model_path = "./outputs/{}.h5".format(model_name)

    os.makedirs('outputs', exist_ok=True)
    model.save(model_path)

    # Log data to Azure experiment
    run.log_list("loss", hist.history["loss"])
    run.log_list("accuracy", hist.history["acc"])
    run.log("test-accuracy", acc)

    return model_name, model_path