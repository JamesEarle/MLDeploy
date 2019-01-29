import requests
import random
import json

import numpy as np
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# http://<your-service-ip>:80/score
uri = "http://104.45.178.103:80/score"

# Take random sample from test set.
index = random.randint(0,len(X_test)-1)
img = X_test[index].reshape(-1)

# Offset 0 base with + 1 
print("Actual label: ", y_test[index] + 1)

input_json = json.dumps({"data": np.asarray(img).tolist() })

res = requests.post(uri, input_json, headers={ "Content-Type": "application/json" })

print(res.text)