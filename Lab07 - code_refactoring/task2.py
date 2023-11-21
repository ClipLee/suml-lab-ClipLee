
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import pickle


def save_data_and_train_model(x, y): # przyjmuje dowolne wartosci x i y

    # zapisanie danych do pliku CSV
    data = np.column_stack((x, y))
    df = pd.DataFrame(data, columns=['x', 'y'])
    df.to_csv('10_points.csv', mode='a', index=False, header=False)

    # wczytanie danych treningowych
    dataset = pd.read_csv('10_points.csv')

    x_train = df['x'].values.reshape(-1, 1)
    y_train = df['y'].values.reshape(-1, 1)

    # trenowanie modelu
    model = LinearRegression()
    model.fit(x_train, y_train)

    print('y = a*x +b')
    print('a = ', model.coef_[0][0])
    print('b = ', model.intercept_[0])

    # zapisanie nowego modelu
    print('...Saving model...')
    pickle.dump(model, open('Lab07 - code_refactoring/our_model.pkl', 'wb'))


# Przykładowe wywołanie funkcji
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
save_data_and_train_model(x, y)
