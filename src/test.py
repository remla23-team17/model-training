"""
Test class for ML pipeline
"""
import pickle
import sys
import joblib

import pandas as pd


def predict(data):
    """
    Predict the labels for the given data using the trained classifier.

    Args:
        data (str or pd.DataFrame): Input data to predict labels for. If a string is provided,
            it will be treated as a single text sample. If a DataFrame is provided, the "Review"
            column will be used as the input data.

    """
    classifier = joblib.load('output/model')
    with open('output/bow.pkl', 'rb') as file:
        cv = pickle.load(file)

    if isinstance(data, str):
        data = [data]
    else:
        data = data["Review"]

    X = cv.transform(data).toarray()
    y_pred = classifier.predict(X)

    print(y_pred)


def main():
    """
    Main entry point of the testing script. It expects a command-line argument specifying the
    path to the input file. The labels are predicted for the input data.

    """
    if len(sys.argv) != 2:
        print("Usage: python test.py <input_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    dataset = pd.read_csv(file_path,
                          delimiter='\t',
                          quoting=3,
                          dtype={'Review': object, 'Liked': int})[:]
    predict(dataset)


if __name__ == "__main__":
    main()
