import os
import json
import numpy as np
from PIL import Image

from keras.models import load_model

from azureml.core.model import Model

classes = {
    0: "axes",
    1: "boots",
    2: "carabiners",
    3: "crampons",
    4: "gloves",
    5: "hardshell_jackets",
    6: "harnesses",
    7: "helmets",
    8: "insulated_jackets",
    9: "pulleys",
    10: "rope",
    11: "tents"
}

def init():
    global model 

    # Path needs .h5 when local, unsure when remote?
    model_path = Model.get_model_path(model_name="keras-gear-cnn-0.8418.h5")
    model = load_model(model_path)

def run(raw_data):
    data = np.array(json.loads(raw_data)['data'])
    return json.dumps(model.predict(np.asarray([data])).tolist())

# def pad_and_resize(img):
#     img = np.asarray(img) # just in case data isn't numpy array
#     try:
#         h = img.shape[0]
#         w = img.shape[1] 
#         pad = abs(h-w)
#         p = int(pad/2)
#     except:
#         print("Image data shape calculation hit an error on variable img of type {}.".format(type(img)))

#     if(h>w): # add to width (columns)
#         padded_img = np.full((h,w+pad,3), 255, dtype=np.uint8)
#         padded_img[:h,p:w+p] = img
#         return Image.fromarray(padded_img,"RGB").resize((128,128))
#     elif(h<w): # add to height (rows)
#         padded_img = np.full((h+pad,w,3), 255, dtype=np.uint8)
#         padded_img[p:h+p,:w] = img
#         return Image.fromarray(padded_img,"RGB").resize((128,128))
#     else:
#         img = np.asarray(img)
#         return Image.fromarray(np.asarray(img),"RGB").resize((128,128))
