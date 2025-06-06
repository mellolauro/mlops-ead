import pandas as pd
import pytest
from tensorflow.keras.models import Sequential

from train import (read_data,
                create_model,
                train_model)


@pytest.fixture
def sample_data():

    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [6, 7, 8, 9, 10],
        'fetal_health': [1, 1, 2, 3, 2]
    })
    return data


def test_read_data():

    X, y = read_data()

    assert not X.empty
    assert not y.empty


def test_create_model():

    X, _ = read_data()
    model = create_model(X)

    assert len(model.layers) > 2
    assert model.trainable
    assert isinstance(model, Sequential)


def test_train_model(sample_data):


    X = sample_data.drop(['fetal_health'], axis=1)
    y = sample_data['fetal_health'] - 1
    model = create_model(X)
    train_model(model, X, y, is_train=False)
    assert model.history.history['loss'][-1] > 0
    assert model.history.history['val_loss'][-1] > 0
