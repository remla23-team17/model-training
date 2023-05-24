import unittest
from pipeline import train


class PerformanceTest(unittest.TestCase):

    def test_accuracy_comparison(self):

        training_data_path = "data/preproccesed_HistoricDump.tsv"

        # Run the first training
        accuracy1 = train.main(training_data_path, 0)

        # Run the second training
        accuracy2 = train.main(training_data_path, 1)

        # Compare the accuracies
        self.assertEqual(accuracy1, accuracy2, "Accuracy values do not match.")