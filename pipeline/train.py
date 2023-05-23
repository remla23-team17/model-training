import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

import joblib
import pickle


def train_model(dataset):
    cv = CountVectorizer(max_features=1420)

    X = cv.fit_transform(dataset["Review"]).toarray()
    y = dataset.iloc[:, -1].values

    __create_output_dir()

    __save_bow(cv)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    joblib.dump(classifier, 'output/c2_Classifier_Sentiment_Model')

    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    return cm, acc


def __create_output_dir():
    cur_directory = os.getcwd()
    output_dir = os.path.join(cur_directory, r'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def __save_bow(cv):
    bow_path = 'output/c1_BoW_Sentiment_Model.pkl'
    pickle.dump(cv, open(bow_path, "wb"))
