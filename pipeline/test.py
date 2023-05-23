import pickle
import joblib


def predict(dataset):
    classifier = joblib.load('output/model')
    cv = pickle.load(open('output/bow.pkl', 'rb'))

    if isinstance(dataset, str):
        dataset = [dataset]
    else:
        dataset = dataset["Review"]

    X = cv.transform(dataset).toarray()
    y_pred = classifier.predict(X)

    return y_pred

