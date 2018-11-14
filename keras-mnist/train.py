# from azureml.core import Run

# def build_train_model():

#     # Loading sample data from within Keras, here you would link to your actual data set locally or in Azure
#     (X_train, y_train), (X_test, y_test) = mnist.load_data()

#     # Simple dense model
#     model = models.Sequential()
#     model.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
#     model.add(layers.Dense(10, activation='softmax'))

#     model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["accuracy"])

#     # Flatten image data, current shape is (nsamples, 28, 28), we want (nsamples, 28*28,)
#     X_train = X_train.reshape((len(X_train), 28*28))
#     X_test = X_test.reshape((len(X_test), 28*28))

#     # Then we normalize our channel data.
#     X_train = X_train.astype('float32') / 255
#     X_test = X_test.astype('float32') / 255

#     # One-hot encoding training and testing labels
#     y_train = to_categorical(y_train)
#     y_test = to_categorical(y_test)

#     # Train on our data
#     model.fit(X_train, y_train, epochs=5, batch_size=128)

#     # Evaluate loss and accuracy
#     metrics = model.evaluate(X_test, y_test)

#     return model, metrics

# def build_train_sklearn_model(): 
from sklearn.linear_model import LogisticRegression
from azureml.core import ScriptRunConfig
from azureml.core import Run
from azureml.core import Workspace
from azureml.core import RunConfiguration
from azureml.core import Experiment

import os
import numpy as np

from keras import models
from keras import layers
from keras.datasets import mnist
from keras.utils import to_categorical

sub_key = os.getenv("AZURE_SUBSCRIPTION")
ws_name = "mldeployworkspace"
skl = False
exp_name = "sklearn-mnist" if skl else "keras-mnist"

# Get workspace
ws = Workspace.get(ws_name, subscription_id=sub_key)

# Get experiment
exp = Experiment(workspace=ws, name=exp_name)
# print(exp)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Flatten image data, current shape is (nsamples, 28, 28), we want (nsamples, 28*28,)
X_train = X_train.reshape((len(X_train), 28*28))
X_test = X_test.reshape((len(X_test), 28*28))

# Then we normalize our channel data.
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

config = ScriptRunConfig(source_directory='.', script='train.py', run_config=RunConfiguration())
run = exp.submit(config)
# print(run.experiment)

if(skl): # temporary, abstract out later
    clf = LogisticRegression(random_state=42)
    print(X_train.shape, y_train.shape)
    clf.fit(X_train, y_train)
    acc = np.average(clf.predict(X_test) == y_test)
    print(acc)
    run.log("accuracy", np.float(acc))
else: #keras
    # One-hot encoding training and testing labels
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model = models.Sequential()
    model.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["accuracy"])

    model.fit(X_train, y_train, epochs=5, batch_size=128)

    # os.makedirs('outputs', exist_ok=True)
    # model.save("./outputs/keras-mnist-model.h5")

    run.log_list("Metric labels: ", model.metrics_names)
    run.log_list("accuracy", model.evaluate(X_test, y_test))
# build_train_sklearn_model()