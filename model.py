from keras import models
from keras import layers

from keras.datasets import mnist
from keras.utils import to_categorical

# Use to grab a previously saved model (.h5 file)
from keras.models import load_model

def save_model(name, model):
    model.save(name + ".h5")

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

# Train on our data
model.fit(X_train, y_train, epochs=5, batch_size=128)

# Save locally
save_model("keras-mnist", model)

# Evaluate model
model.evaluate(X_test, y_test)