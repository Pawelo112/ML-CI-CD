import numpy as np
from model import train_and_predict, get_accuracy


def test_predictions_not_none():
    """
    Test 1: Sprawdza, czy otrzymujemy jakąkolwiek predykcję.
    """
    preds, _ = train_and_predict()
    assert preds is not None, "Predictions should not be None."


def test_predictions_length():
    """
    Test 2: Sprawdza, czy długość listy predykcji jest większa od 0 i czy odpowiada przewidywanej liczbie próbek testowych.
    """
    preds, y_test = train_and_predict()
    assert len(preds) > 0, "Predictions should not be empty."
    assert len(preds) == len(y_test), "Predictions and test labels must have the same length."


def test_predictions_value_range():
    """
    Test 3: Sprawdza, czy wartości w predykcjach mieszczą się w spodziewanym zakresie:
    Dla zbioru Iris mamy 3 klasy (0, 1, 2).
    """
    preds, _ = train_and_predict()
    assert all(pred in [0, 1, 2] for pred in preds), "Predictions should only contain class labels 0, 1, or 2."


def test_model_accuracy():
    """
    Test 4: Sprawdza, czy model osiąga co najmniej 70% dokładności.
    """
    preds, y_test = train_and_predict()

    # Pobieramy wartość dokładności za pomocą stworzonej metody
    acc = get_accuracy(preds, y_test)

    assert acc >= 0.7, f'Model accuracy should be >= 70%. Instead got {acc:.2f}'

