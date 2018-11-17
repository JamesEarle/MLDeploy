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

    # delete
    ws = Workspace.get("mldeployworkspace", subscription_id=sub_key)

    # retreive the path to the model file using the model name
    model_path = Model.get_model_path('keras-mnist-9815', _workspace=ws)
    model = load_model(model_path)
    print(model.name)

def run(raw_data):
    data = np.array(json.loads(raw_data)['data'])
    # make prediction
    y_hat = model.predict(data)
    return json.dumps(y_hat.tolist())