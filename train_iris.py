# save_model.py
import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import os

iris = load_iris()
X, y = iris.data, iris.target

model = RandomForestClassifier()
model.fit(X, y)


filename = "app/model_iris.pkl"
os.makedirs(os.path.dirname(filename), exist_ok=True)
if not os.path.isfile(filename):
    with open(filename, "wb") as f:
        pickle.dump(model, f)
