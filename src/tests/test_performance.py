"""
Pytest Suite for model performance
"""

import pytest
from src import train
from src import preprocess


@pytest.fixture(name="preprocess_data")
def fixture_preprocess_data():
    """
    Fixture that loads and preprocesses the training dataset.

    Returns:
        pd.DataFrame: Preprocessed training dataset.

    """
    train_dataset = preprocess.load_data('data/HistoricDump.tsv')
    return train_dataset


def test_train_accuracy_comparison(preprocess_data):
    """
    Test the accuracy comparison between two training runs.

    Args:
        preprocess_data (pd.DataFrame): Preprocessed training dataset.

    """
    _, accuracy1 = train.train_model(preprocess_data, 0)
    _, accuracy2 = train.train_model(preprocess_data, 1)
    assert abs(accuracy1 - accuracy2) <= 0.1
