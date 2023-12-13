from inspect import signature
from json import load
from pyexpat import model
from unittest import result
import mlflow
from mlflow.models import infer_signature
from numpy import sign

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

# To launch the server: `mlflow server --host 127.0.0.1 --port 8080`

dsp6 = pd.read_csv('Lab10/data/DSP_6_Clean.csv')
print(dsp6)

x = dsp6.drop(['Survived'], axis=1)
y = dsp6['Survived']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.10, random_state=101)

params_forest = {
    'n_estimators': 10,
    'random_state': 0,
}

params_lreg = {
    "max_iter": 500,
    "random_state": 0,
    "solver": "lbfgs",
}

# training
forest = RandomForestClassifier(**params_forest)
forest.fit(x_train, y_train)
score_forest = forest.score(x_train, y_train)

lreg = LogisticRegression(**params_lreg)
lreg.fit(x_train, y_train)
score_lreg = lreg.score(x_train, y_train)

tree = DecisionTreeClassifier()
tree.fit(x_train, y_train)
score_tree = tree.score(x_train, y_train)

# prediction
y1_predict = forest.predict(x_test)
y2_predict = lreg.predict(x_test)
y3_predict = tree.predict(x_test)

# metrics
acc_forest = accuracy_score(y_test, y1_predict)
acc_lreg = accuracy_score(y_test, y2_predict)
acc_tree = accuracy_score(y_test, y3_predict)

# server address settings

mlflow.set_tracking_uri("http://localhost:8080")
# mlflow.set_tracking_uri("http://127.0.0.1:8080")

# experiment settings
mlflow.set_experiment("Mlflow Titanic")

# run mlflow
with mlflow.start_run():
    # register hyperparameters
    mlflow.log_params(params_forest)
    mlflow.log_params(params_lreg)

    # register acc
    mlflow.log_metric("acc_forest", acc_forest)
    mlflow.log_metric("acc_lreg", acc_lreg)
    mlflow.log_metric("acc_tree", acc_tree)

    # tag settings
    mlflow.set_tag(
        "training info", "standard models: random forests, logistic regression, decision tree for Titanic data")

    # tags settings to descrite results
    sign_forest = infer_signature(x_train, y1_predict)
    sign_lreg = infer_signature(x_train, y2_predict)
    sign_tree = infer_signature(x_train, y3_predict)

    # register models
    model_info_forest = mlflow.sklearn.log_model(
        sk_model=forest,
        artifact_path="titanic_model_forest",
        signature=sign_forest,
        input_example=x_train,
        registered_model_name="titanic_ml_forest"
    )

    model_info_lreg = mlflow.sklearn.log_model(
        sk_model=lreg,
        artifact_path="titanic_model_lreg",
        signature=sign_lreg,
        input_example=x_train,
        registered_model_name="titanic_ml_lreg"
    )
    
    model_info_tree = mlflow.sklearn.log_model(
        sk_model=tree,
        artifact_path="titanic_model_tree",
        signature=sign_tree,
        input_example=x_train,
        registered_model_name="titanic_ml_tree"
    )

# load model
loaded_model_forest = mlflow.pyfunc.load_model(model_info_forest.model_uri)
loaded_model_lreg = mlflow.pyfunc.load_model(model_info_lreg.model_uri)
loaded_model_tree = mlflow.pyfunc.load_model(model_info_tree.model_uri)

pred1 = loaded_model_forest.predict(x_test)
pred2 = loaded_model_lreg.predict(x_test)
pred3 = loaded_model_tree.predict(x_test)

titanic_acc = ["Random Forest", "Logistic Regression", "Decision Tree"]

result = pd.DataFrame()
result[titanic_acc[0]] = pred1
result[titanic_acc[1]] = pred2
result[titanic_acc[2]] = pred3
