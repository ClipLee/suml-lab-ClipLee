import numpy as np
import pandas as pd
import pickle


def predict_y(x):
    imported_model = pickle.load(
        open('Lab07 - code_refactoring/our_model.pkl', 'rb'))

    x = np.array([2.78])
    y = imported_model.predict(x.reshape(-1, 1))
    return y


# # test
# x = np.array([[1, 2, 3]])  # jaki input
# predicted_y = predict_y(x)
# print(predicted_y)
