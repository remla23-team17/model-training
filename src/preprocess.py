"""
Preprocessor class for ML pipeline
"""

import os
import re
import sys

import nltk

import pandas as pd

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')


def load_data(path):
    """
    Load the dataset from the specified path and preprocess the 'Review' column.

    Args:
        path (str): The path to the dataset file.

    Returns:
        pd.DataFrame: The preprocessed dataset.

    """
    dataset = pd.read_csv(path,
                          delimiter='\t',
                          quoting=3,
                          dtype={'Review': object, 'Liked': int})[:]

    dataset['Review'] = dataset['Review'].apply(__create_corpus)
    return dataset


def __create_corpus(review):
    """
    Preprocess the given review by removing non-alphabetic characters, converting to lowercase,
    tokenizing, removing stopwords, and applying stemming.

    Args:
        review (str): The review to preprocess.

    Returns:
        str: The preprocessed review.

    """
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    return review


def preprocess_data(path):
    """
    Preprocess the data from the specified file by loading the
    dataset and saving the preprocessed data to a new file.

    Args:
        path (str): The path to the dataset file.

    """
    file_name = os.path.basename(path)
    raw_data = load_data(path)
    raw_data.to_csv(f'data/preprocessed_{file_name}', sep="\t", index=False)


def main():
    """
    Main entry point of the preprocessing script. It expects a command line argument specifying the
    path to the input file. The data is preprocessed and saved to a new file.

    """
    if len(sys.argv) != 2:
        print("Usage: python preprocess.py <input_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    preprocess_data(file_path)


if __name__ == "__main__":
    main()
