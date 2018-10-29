from keras import models
from keras import layers

from keras.datasets import mnist
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# Flatten image data, current shape is (nsamples, 28, 28), we want (nsamples, 28*28,)
train_images = train_images.reshape((len(train_images), 28*28))
test_images = test_images.reshape((len(test_images), 28*28))

# Then we normalize our channel data.
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# One-hot encoding training and testing labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.fit(train_images, train_labels, epochs=5, batch_size=128)