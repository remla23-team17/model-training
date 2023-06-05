"""
Train class for ML pipeline
"""
import os
import sys
import json
import pickle
import joblib

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def create_json_dump(cm, acc):
    """
    Create a JSON dump of the confusion matrix and accuracy.

    Args:
        cm (numpy.ndarray): Confusion matrix.
        acc (float): Accuracy.

    """
    tn, fp, fn, tp = cm.ravel()
    data = {
        "TP_Count": int(tp),
        "TN_Count": int(tn),
        "FP_Count": int(fp),
        "FN_Count": int(fn),
        "accuracy": acc
    }

    json_data = json.dumps(data)

    with open("output/performance.json", "w", encoding="utf-8") as f:
        f.write(json_data)


def train_model(data, seed):
    """
    Train a Naive Bayes classifier using the given data and save the model.

    Args:
        data (pd.DataFrame): Input dataset.
        seed (int): Random seed for train-test split.

    Returns:
        numpy.ndarray: Confusion matrix.
        float: Accuracy.

    """
    cv = CountVectorizer(max_features=1420)

    X = cv.fit_transform(data["Review"].astype('U')).toarray()
    y = data.iloc[:, -1].values

    __create_output_dir()

    __save_bow(cv)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)

    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    joblib.dump(classifier, 'output/model')

    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    create_json_dump(cm, acc)
    print(cm)
    print(acc)
    return cm, acc


def __create_output_dir():
    """
    Create the 'output' directory if it doesn't exist.

    """
    cur_directory = os.getcwd()
    output_dir = os.path.join(cur_directory, r'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def __save_bow(cv):
    """
    Save the CountVectorizer object as a pickle file.

    Args:
        cv (CountVectorizer): CountVectorizer object.

    """
    bow_path = 'output/bow.pkl'
    with open(bow_path, "wb") as file:
        pickle.dump(cv, file)


def main():
    """
    Main entry point of the training script. It expects a command-line argument specifying the
    path to the input file. Additionally, an optional second argument can be provided as the
    random seed for train-test split.

    """
    if len(sys.argv) != 2:
        print("Usage: python train.py <input_file>")
        sys.exit(1)

    file_path = sys.argv[1]

    if len(sys.argv) == 3:
        input_seed = int(sys.argv[2])
    else:
        input_seed = 0

    dataset = pd.read_csv(file_path,
                          delimiter='\t',
                          quoting=3,
                          dtype={'Review': object, 'Liked': int})[:]

    train_model(dataset, input_seed)


if __name__ == "__main__":
    main()
