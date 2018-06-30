"""feature generation."""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import hstack, csr_matrix
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, Lasso, SGDRegressor, HuberRegressor
stop = set(stopwords.words('russian'))
plt.style.use('ggplot')
from nltk.tokenize import TweetTokenizer
tok = TweetTokenizer()

class MetaFeaturesGenerator(object):
    """Process data for Avito competition."""

    def __init__(self, folder=''):
        """Init."""
        self.train = pd.read_csv(os.path.join(folder, 'train.csv.zip'), compression='zip')
        self.test = pd.read_csv(os.path.join(folder, 'test.csv.zip'), compression='zip')
        self.models = [Ridge(), SGDRegressor(max_iter=100, alpha=0.00001, eta0=0.1), Lasso(), HuberRegressor()]
        self.kf = KFold(n_splits=5)
        self.folder_name = 'pickles'

    def vectorize(self, ngram_range=(1, 3)):
        """
        Vectorize texts.

        Vectorizers for title, desctiption and params are different

        """
        self.y = self.train['deal_probability']

        # dropping unnecessary colunbs
        self.train = self.train[['title', 'description', 'param_1', 'param_2', 'param_3', 'deal_probability']]
        self.test = self.test[['title', 'description', 'param_1', 'param_2', 'param_3']]

        self.train['description'].fillna('', inplace=True)
        self.train['description'] = self.train['description'].astype(str)
        self.train['description'] = self.train['description'].apply(lambda x: str(x).replace('/\n', ' ').replace('\xa0', ' ').replace('.', '. ').replace(',', ', '))
        self.test['description'].fillna('', inplace=True)
        self.test['description'] = self.test['description'].astype(str)
        self.test['description'] = self.test['description'].apply(lambda x: str(x).replace('/\n', ' ').replace('\xa0', ' ').replace('.', '. ').replace(',', ', '))
        self.train['title'].fillna('', inplace=True)
        self.train['title'] = self.train['title'].astype(str)
        self.test['title'].fillna('', inplace=True)
        self.test['title'] = self.test['title'].astype(str)

        self.train['params'] = self.train['param_1'].fillna('').astype(str) + ' ' + self.train['param_2'].fillna('').astype(str) + ' ' + self.train['param_3'].fillna('').astype(str)
        self.train['params'] = self.train['params'].str.strip()

        self.test['params'] = self.test['param_1'].fillna('').astype(str) + ' ' + self.test['param_2'].fillna('').astype(str) + ' ' + self.test['param_3'].fillna('').astype(str)
        self.test['params'] = self.test['params'].str.strip()

        self.ngram_name = str(ngram_range[0]) + '_' + str(ngram_range[1])
        self.title_vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=5, max_features=10000, tokenizer=tok.tokenize)
        self.title_vectorizer.fit(self.train['title'])
        self.train_title = self.title_vectorizer.transform(self.train['title'])
        self.test_title = self.title_vectorizer.transform(self.test['title'])

        self.description_vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_df=0.9, min_df=5,
                                                      max_features=50000, tokenizer=tok.tokenize)
        self.description_vectorizer.fit(self.train['description'])
        self.train_description = self.description_vectorizer.transform(self.train['description'])
        self.test_description = self.description_vectorizer.transform(self.test['description'])

        self.params_vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_df=0.9, min_df=5,
                                                 max_features=50000, tokenizer=tok.tokenize)
        self.params_vectorizer.fit(self.train['params'])
        self.train_params = self.params_vectorizer.transform(self.train['params'])
        self.test_params = self.params_vectorizer.transform(self.test['params'])

        self.X_texts = csr_matrix(hstack([self.train_title, self.train_description, self.train_params]))
        self.X_test_texts = csr_matrix(hstack([self.test_title, self.test_description, self.test_params]))

        path_name = self.folder_name + '/'
        with open(path_name + 'X_texts_tweet.pickle', 'wb') as f:
            pickle.dump(self.X_texts, f)
        with open(path_name + 'X_test_texts_tweet.pickle', 'wb') as f:
            pickle.dump(self.X_test_texts, f)

    def make_meta_features(self):
        """
        Make meta features.

        Train one or several classifiers and create meta features.
        """
        
        for model in self.models:
            X_meta = np.zeros((1503424, 1))
            X_test_meta = []
            for fold_i, (train_i, test_i) in enumerate(self.kf.split(self.X_texts)):
                clf = model
                clf.fit(self.X_texts.tocsr()[train_i], self.y[train_i])
                X_meta[test_i, :] = clf.predict(self.X_texts.tocsr()[test_i]).reshape(-1, 1)
                X_test_meta.append(clf.predict(self.X_test_texts))
            X_test_meta = np.stack(X_test_meta)
            X_test_meta_mean = np.mean(X_test_meta, axis=0)
            train_path = self.folder_name + '/' + self.ngram_name + '/' + 'X_meta_tweet' + str(model).split('(')[0].lower() + '.pickle'

            with open(train_path, 'wb') as f:
                pickle.dump(X_meta, f)

            test_path = self.folder_name + '/' + self.ngram_name + '/' + 'X_test_meta_mean_tweet' + str(model).split('(')[0].lower() + '.pickle'
            with open(test_path, 'wb') as f:
                pickle.dump(X_test_meta_mean, f)
