import pickle
import joblib


def predict(dataset):
    classifier = joblib.load('output/c2_Classifier_Sentiment_Model')
    cv = pickle.load(open('output/c1_BoW_Sentiment_Model.pkl', 'rb'))

    if isinstance(dataset, str):
        dataset = [dataset]
    else:
        dataset = dataset["Review"]

    X = cv.transform(dataset).toarray()
    y_pred = classifier.predict(X)

    return y_pred

