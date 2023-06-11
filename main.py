import src.preprocess as preprocessor
import src.train as train
import src.production as model


def execute_pipeline():
    train_dataset = preprocessor.load_data('data/HistoricDump.tsv')
    train.train_model(train_dataset, 0)

    test_dataset = preprocessor.load_data('data/FreshDump.tsv')
    model.predict(test_dataset)

    test_review = "Their regular toasted bread was equally satisfying with the occasional pats of butter... Mmmm..."
    model.predict(test_review)


if __name__ == '__main__':
    execute_pipeline()
