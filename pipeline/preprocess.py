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
    dataset = pd.read_csv(path, delimiter='\t', quoting=3)
    dataset['Review'] = dataset['Review'].apply(__create_corpus)
    return dataset


def __create_corpus(review):
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    return review


def preprocess_data(path):
    file_name = os.path.basename(path)
    raw_data = load_data(path)
    raw_data.to_csv(f'data/preproccesed_{file_name}', sep="\t", index=False)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python preprocess.py <input_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    preprocess_data(file_path)
