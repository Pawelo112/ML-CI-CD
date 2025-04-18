import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Tworzymy przykładowy model klasyfikacyjny z iris_dataset,
# w celu wykonania przykładowych testów (plik test_model.py)
def train_and_predict():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Zwracamy predykcje i zbiór testowy
    return predictions, y_test


# Zwracamy dokładność modeli
def get_accuracy(predictions, y_test):
    return accuracy_score(y_test, predictions)
