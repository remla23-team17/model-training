import pickle
import sys

import joblib
import pandas as pd


def predict(data):
    classifier = joblib.load('output/model')
    cv = pickle.load(open('output/bow.pkl', 'rb'))

    if isinstance(data, str):
        data = [data]
    else:
        data = data["Review"]

    X = cv.transform(data).toarray()
    y_pred = classifier.predict(X)

    print(y_pred)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test.py <input_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    dataset = pd.read_csv(file_path, delimiter='\t', quoting=3)
    predict(dataset)
