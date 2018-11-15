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

rg = "MLDeploy"
loc = "eastus2"
sub_key = os.getenv("AZURE_SUBSCRIPTION")
ws_name = "mldeployworkspace"
exp_name = "keras-mnist"

# Get or create workspace
try: # Try to get, if it fails create a new one
    ws = Workspace.get(ws_name, subscription_id=sub_key)
except:
    ws = Workspace.create(name=ws_name,
                          subscription_id=sub_key,
                          resource_group=rg,
                          create_resource_group=False,
                          location=loc)

# Get experiment
exp = Experiment(workspace=ws, name=exp_name)

run = exp.start_logging()

# Load in training data from predefined Keras dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Flatten image data, current shape is (nsamples, 28, 28), we want (nsamples, 28*28,)
X_train = X_train.reshape((len(X_train), 28*28))
X_test = X_test.reshape((len(X_test), 28*28))

# Then we normalize our channel data.
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# One-hot encoding training and testing laSbels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Simple dense model
model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["accuracy"])

# Access history of loss and accuracy across all epochs
hist = model.fit(X_train, y_train, epochs=5, batch_size=128)

# Save model locally
os.makedirs('outputs', exist_ok=True)
model.save("./outputs/keras-mnist-model.h5")

# Log data to Azure experiment
run.log_list("loss", hist.history["loss"])
run.log_list("accuracy", hist.history["acc"])

# Create an estimator here and submit web job to batch AI cluster before registering model?
# or only if we want to train on the compute target?

acc = model.evaluate(X_test, y_test)
acc = str(acc[1])[2:] # index to accuracy, slice out "0." and only use RHS of float, "9815"

# Register your model in Azure
model = run.register_model(model_name="keras-mnist-" + acc, model_path="outputs/keras-mnist-model.h5")
print(model.name, model.id, model.version, sep = '\t')

# Complete the run
run.complete()


# Print URL to show this experiment run in Azure portal.
print(run.get_portal_url())