from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import pickle as pk
import os

# TODO stworzyć bardziej uniwersalny i przydatny model z wczytywaniem ścieżek domyślnej i użytkownika


def save_data_and_train_model(x, y, model_path=None, file_name=None):
    # def save_data_and_train_model(x=[1, 2, 3], y=[4, 5, 6], model_path=None):

    # korzystanie z domyślnej nazwy
    default_file_name = '10_points.csv'

    # jeśli użytkownik nie podał nazwy pliku, użyj domyślnej
    if file_name is None:
        file_name = default_file_name

    # zapisanie danych do pliku CSV
    data = np.column_stack((x, y))
    df = pd.DataFrame(data, columns=['x', 'y'])
    df.to_csv('10_points.csv', mode='a', index=False, header=False)

    # wczytanie danych treningowych
    # TODO czy jest to potrzebne, skoro zawiera się to w ścieżce?
    dataset = pd.read_csv(file_name)

    x_train = df['x'].values.reshape(-1, 1)
    y_train = df['y'].values.reshape(-1, 1)

    # domyślna ścieżka do modelu
    default_model_path = 'Lab07 - code_refactoring/our_model.pkl'

    # jeśli użytkownik nie podał ścieżki, użyj domyślnej
    if model_path is None:
        model_path = default_model_path

    # wczytanie poprzedniego i trenowanie modelu
    if os.path.exists(model_path):
        # wczytanie istniejącego modelu
        print('✔     Model was found     ✔')
        print('...Loading existing model...\n')
        model = pk.load(open(model_path, 'rb'))
    else:
        # trenowanie nowego modelu
        print('❌..Model not found..❌')
        print('...Training new model...')
        model = LinearRegression()
        model.fit(x_train, y_train)

    model.fit(x_train, y_train)

    print('y = a * x + b')
    print('a = ', model.coef_[0][0])
    print('b = ', model.intercept_[0])

    # zapisanie nowego modelu
    print('\n...  Saving model  ...')
    pk.dump(model, open('Lab07 - code_refactoring/our_model.pkl', 'wb'))
    print('✔     Model saved     ✔')


# # wywołanie funkcji
# x = np.array([1, 2, 3])
# y = np.array([4, 5, 6])
# save_data_and_train_model(x, y)
