import pipeline.preprocess as preprocessor
import pipeline.train as train
import pipeline.test as test


def execute_pipeline():
    train_dataset = preprocessor.load_data('data/HistoricDump.tsv')
    train.train_model(train_dataset)

    test_dataset = preprocessor.load_data('data/FreshDump.tsv')
    test.predict(test_dataset)

    test_review = "Their regular toasted bread was equally satisfying with the occasional pats of butter... Mmmm..."
    test.predict(test_review)


if __name__ == '__main__':
    execute_pipeline()
