import pipeline.preprocess as preprocessor
import pipeline.train as train
import pipeline.test as test


def execute_pipeline():
    train_dataset = preprocessor.load('datasets/a1_RestaurantReviews_HistoricDump.tsv')
    training_performance = train.train_model(train_dataset)
    print(training_performance)

    test_dataset = preprocessor.load('datasets/a2_RestaurantReviews_FreshDump.tsv')
    predictions = test.test_model(test_dataset)
    print(predictions)


if __name__ == '__main__':
    execute_pipeline()
