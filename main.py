import pipeline.preprocess as preprocessor
import pipeline.train as train
import pipeline.test as test


def execute_pipeline():
    train_dataset = preprocessor.load_data('data/a1_RestaurantReviews_HistoricDump.tsv')
    training_performance = train.train_model(train_dataset)
    print(training_performance)

    test_dataset = preprocessor.load_data('data/a2_RestaurantReviews_FreshDump.tsv')
    predictions = test.predict(test_dataset)
    print(predictions)

    test_review = "Their regular toasted bread was equally satisfying with the occasional pats of butter... Mmmm..."
    predictions = test.predict(test_review)
    print(predictions)


if __name__ == '__main__':
    execute_pipeline()
