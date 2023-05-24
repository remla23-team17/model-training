import os
import sys
import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

import joblib
import pickle


def create_json_dump(cm, acc):
    TN, FP, FN, TP = cm.ravel()
    data = {
        "TP_Count": int(TP),
        "TN_Count": int(TN),
        "FP_Count": int(FP),
        "FN_Count": int(FN),
        "accuracy": acc
    }

    json_data = json.dumps(data)

    with open("output/performance.json", "w") as f:
        f.write(json_data)


def train_model(data):
    cv = CountVectorizer(max_features=1420)

    X = cv.fit_transform(data["Review"].values.astype('U')).toarray()
    y = data.iloc[:, -1].values

    __create_output_dir()

    __save_bow(cv)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    joblib.dump(classifier, 'output/model')

    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    create_json_dump(cm, acc)
    print(cm)
    print(acc)


def __create_output_dir():
    cur_directory = os.getcwd()
    output_dir = os.path.join(cur_directory, r'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def __save_bow(cv):
    bow_path = 'output/bow.pkl'
    pickle.dump(cv, open(bow_path, "wb"))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train.py <input_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    dataset = pd.read_csv(file_path, delimiter='\t', quoting=3)
    train_model(dataset)