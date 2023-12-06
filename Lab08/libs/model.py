import pickle
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression


def predict(x, ml_model):
    x = np.array([x]).reshape(-1, 1)
    model = pickle.load(open(ml_model, "rb"))
    y_pred = model.predict(x)
    return y_pred


def train(x, y, path2csv, path2pickle):
    df = pd.read_csv(path2csv)
    df.loc[len(df.index)] = [x, y]
    df.to_csv(path2csv, index=False)

    X = df["x"].values.reshape(-1, 1)
    y = df['y'].values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, y)

    pickle.dump(model, open(path2pickle, 'wb'))
