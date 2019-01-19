import json
import numpy as np
import os
import pickle

from azureml.core.model import Model
from azureml.core import Workspace

# Use to grab a previously saved model (.h5 file)
from keras.models import load_model
from keras.datasets import mnist

sub_key = os.getenv("AZURE_SUBSCRIPTION")

def init():
    global model 

    # Update model name based on your own model file
    model_path = Model.get_model_path(model_name="keras-mnist-9794.h5")
    model = load_model(model_path)

def run(raw_data):
    data = np.array(json.loads(raw_data)['data'])
    return json.dumps(model.predict(np.asarray([data])).tolist())