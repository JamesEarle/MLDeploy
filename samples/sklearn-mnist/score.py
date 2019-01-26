import json
import numpy as np
import os
# import pickle

from azureml.core.model import Model
from azureml.core import Workspace

# Use to grab a previously saved model (.pkl file)
from sklearn.externals import joblib

def init():
    global model 

    # Update model name based on your own model file
    model_path = Model.get_model_path(model_name="sklearn-mnist-9201.pkl")
    model = joblib.load(model_path)

def run(raw_data):
    data = np.array(json.loads(raw_data)['data'])
    return json.dumps(model.predict(np.asarray([data])).tolist())