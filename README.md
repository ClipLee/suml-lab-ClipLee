<!-- https://classroom.github.com/online_ide?assignment_repo_id=12181533&assignment_repo_type=AssignmentRepo -->

# AI & ML Labs

To  repozytorium zawiera różne laboratoria i zadania związane z uczeniem maszynowym i analizą danych, które zostały przeprowadzone w celu nauki i praktyki dziedzin data scienece i machine learning.

Oto przegląd zawartości repozytorium oraz zadań i sposobów ich rozwiązania:

## Struktura Repozytorium

- **Lab03**: Zawiera zestawy danych.
- **Lab05**: Zawiera plik CSV (`DSP_4.csv`) oraz aplikację Streamlit (`streamlit_app.py`).
- **Lab06**: Zawiera aplikację ([`app.py`](Lab07%20-%20code_refactoring/app.py)), model ([`model.h5`](Lab08/libs/model.py)) oraz notebook Jupyter (`streamlit (titanic).ipynb`).
- **Lab07 - code_refactoring**: Zawiera zrefaktoryzowany kod, aplikację ([`app.py`](Lab07%20-%20code_refactoring/app.py)), dane oraz dwa zadania ([`task1.py`](Lab07%20-%20code_refactoring/task1.py), [`task2.py`](Lab07%20-%20code_refactoring/task2.py)).
- **Lab08**: Zawiera aplikację ([`app.py`](Lab07%20-%20code_refactoring/app.py)), dane, biblioteki, modele ML oraz plik `requirements.txt`.
- **Lab10**: Zawiera dane oraz skrypt [`ml_flow.py`](Lab10/ml_flow.py).
- **Lab12**: Zawiera notebooki Jupyter oraz modele ML.
- **Lista_1.ipynb**, **Lista_2.ipynb**, **Regresja.ipynb**: Notebooki Jupyter z różnymi zadaniami.
- **mlartifacts**, **mlruns**: Artefakty i wyniki eksperymentów ML.
- **README.md**: Plik z opisem repozytorium.
- **Zadania**: Folder z zadaniami.

## Przykładowe Zadania i Ich Rozwiązania

### Lab07 - Code Refactoring

- **app.py**: Główna aplikacja, która importuje funkcje z [`task1.py`](Lab07%20-%20code_refactoring/task1.py) i [`task2.py`](Lab07%20-%20code_refactoring/task2.py).

  ```py
  from task1 import predict_y
  from task2 import save_data_and_train_model

  def main():
      # Kod główny aplikacji
      ...

  if __name__ == "__main__":
      main()
  ```

- **task1.py**: Zawiera funkcję `predict_y`, która prawdopodobnie przewiduje wartości na podstawie modelu.
- **task2.py**: Zawiera funkcję `save_data_and_train_model`, która zapisuje dane i trenuje model.

### Lab08

- **app.py**: Prosta aplikacja zwracająca wiadomość.
  
  ```py
  async def index():
      return {"message": "Linear Regression ML"}
  ```

### Lab10

- **ml_flow.py**: Skrypt do trenowania modeli ML (Random Forest, Logistic Regression, Decision Tree) na danych z pliku CSV oraz logowania wyników do MLflow.

  ```py
  import mlflow
  from mlflow.models import infer_signature
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.linear_model import LogisticRegression
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import accuracy_score
  import pandas as pd

  # Wczytanie danych
  dsp6 = pd.read_csv('Lab10/data/DSP_6_Clean.csv')
  x = dsp6.drop(['Survived'], axis=1)
  y = dsp6['Survived']

  # Podział danych na treningowe i testowe
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=101)

  # Parametry modeli
  params_forest = {'n_estimators': 10, 'random_state': 0}
  params_lreg = {"max_iter": 500, "random_state": 0, "solver": "lbfgs"}

  # Trenowanie modeli
  forest = RandomForestClassifier(**params_forest)
  forest.fit(x_train, y_train)
  lreg = LogisticRegression(**params_lreg)
  lreg.fit(x_train, y_train)
  tree = DecisionTreeClassifier()
  tree.fit(x_train, y_train)

  # Predykcje i metryki
  y1_predict = forest.predict(x_test)
  y2_predict = lreg.predict(x_test)
  y3_predict = tree.predict(x_test)
  acc_forest = accuracy_score(y_test, y1_predict)
  acc_lreg = accuracy_score(y_test, y2_predict)
  acc_tree = accuracy_score(y_test, y3_predict)

  # Logowanie wyników do MLflow
  mlflow.set_tracking_uri("http://localhost:8080")
  mlflow.set_experiment("Mlflow Titanic")
  with mlflow.start_run():
      mlflow.log_params(params_forest)
      mlflow.log_params(params_lreg)
      mlflow.log_metric("acc_forest", acc_forest)
      mlflow.log_metric("acc_lreg", acc_lreg)
      mlflow.log_metric("acc_tree", acc_tree)
      sign_forest = infer_signature(x_train, y1_predict)
      sign_lreg = infer_signature(x_train, y2_predict)
      sign_tree = infer_signature(x_train, y3_predict)
      mlflow.sklearn.log_model(forest, "titanic_model_forest", signature=sign_forest)
      mlflow.sklearn.log_model(lreg, "titanic_model_lreg", signature=sign_lreg)
      mlflow.sklearn.log_model(tree, "titanic_model_tree", signature=sign_tree)
  ```

## Podsumowanie

Repozytorium zawiera różne laboratoria i zadania związane z uczeniem maszynowym, analizą danych oraz implementacją modeli ML. Zadania są rozwiązywane poprzez implementację skryptów Python, notebooków Jupyter oraz aplikacji Streamlit. Wyniki i modele są logowane i zarządzane za pomocą MLflow.
