import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os


df = pd.read_csv("Housing.csv")

# preprocess data
df.replace({
    "yes": 1, "no": 0,
    "furnished": 2, "semi-furnished": 1, "unfurnished": 0
}, inplace=True)


X = df.drop("price", axis=1)
y = df["price"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor()
model.fit(X_train, y_train)


filename = "app/model_house.pkl"
os.makedirs(os.path.dirname(filename), exist_ok=True)
if not os.path.isfile(filename):
    with open(filename, "wb") as f:
        pickle.dump(model, f)
