import re
import nltk

import pandas as pd

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')


def load(file_path):
    dataset = pd.read_csv(file_path, delimiter='\t', quoting=3)
    dataset['Review'] = dataset['Review'].apply(create_corpus)
    return dataset


def load_single(input_line):
    return create_corpus(input_line)


def create_corpus(review):
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    return review
