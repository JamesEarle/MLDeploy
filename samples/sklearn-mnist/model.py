# See utils.py for load_data definition.
from utils import load_data
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

# note we also shrink the intensity values (X) from 0-255 to 0-1. This helps the model converge faster.
X_train = load_data('./data/train-images.gz', False) / 255.0
y_train = load_data('./data/train-labels.gz', True).reshape(-1)

X_test = load_data('./data/test-images.gz', False) / 255.0
y_test = load_data('./data/test-labels.gz', True).reshape(-1)

clf = LogisticRegression()
clf.fit(X_train, y_train)

y_hat = clf.predict(X_test)
print(np.average(y_hat == y_test))

joblib.dump(value=clf, filename='sklearn-mnist-9201.pkl')