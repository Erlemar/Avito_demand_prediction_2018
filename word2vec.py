"""Train word2vec."""

import pandas as pd
from gensim.models import Word2Vec
from keras.preprocessing.text import text_to_word_sequence
import logging
import time
from tqdm import tqdm
import datetime


logging.basicConfig(level=logging.INFO)
use_cols = ['param_1', 'param_2', 'param_3', 'title', 'description']


def load_text(start, file):
    """Load data."""
    print('Loading data...', end='')
    tic = time.time()
    if file == 'train':
        train2 = pd.read_csv('train.csv', usecols=use_cols, nrows=1000000, skiprows=range(1, start))
    elif file == 'test':
        train2 = pd.read_csv('test.csv', usecols=use_cols, nrows=1000000, skiprows=range(1, start))
    elif file == 'train_active':
        train2 = pd.read_csv('train_active.csv', usecols=use_cols, nrows=1000000, skiprows=range(1, start))
    elif file == 'test_active':
        train2 = pd.read_csv('test_active.csv', usecols=use_cols, nrows=1000000, skiprows=range(1, start))
    toc = time.time()
    print('Done in {:.1f}s'.format(toc - tic))
    train2['text'] = train2['param_1'].str.cat([train2.param_2, train2.param_3, train2.title,
                                                train2.description], sep=' ', na_rep='')
    train2.drop(use_cols, axis=1, inplace=True)
    train2 = train2['text'].values

    train2 = [text_to_word_sequence(text) for text in tqdm(train2)]
    return train2

model = Word2Vec(size=100, window=10, max_vocab_size=150000000, workers=11, sg=0)
print('Train')
for k in range(15):
    start_time = datetime.datetime.now()
    print('Train', k, 'of', 15)
    update = False
    if k != 0:
        update = True
    train = load_text(k * 1000000 + 1, 'train')
    model.build_vocab(train, update=update)
    model.train(train, total_examples=model.corpus_count, epochs=5)
    end_time = datetime.datetime.now()
    print('Took {0} seconds.'.format((end_time - start_time).seconds))

print('Test')
for k in range(5):
    start_time = datetime.datetime.now()
    print('Test', k, 'of', 5)
    if k != 0:
        update = True
    train = load_text(k * 1000000 + 1, 'test')
    model.build_vocab(train, update=update)
    model.train(train, total_examples=model.corpus_count, epochs=5)
    end_time = datetime.datetime.now()
    print('Took {0} seconds.'.format((end_time - start_time).seconds))

print('Train active')
for k in range(141):
    start_time = datetime.datetime.now()
    print('Train active', k, 'of', 141)
    if k != 0:
        update = True
    train = load_text(k * 1000000 + 1, 'train_active')
    model.build_vocab(train, update=update)
    model.train(train, total_examples=model.corpus_count, epochs=5)
    end_time = datetime.datetime.now()
    print('Took {0} seconds.'.format((end_time - start_time).seconds))

print('Test active')
for k in range(128):
    start_time = datetime.datetime.now()
    print('Test active', k, 'of', 128)
    if k != 0:
        update = True
    train = load_text(k * 1000000 + 1, 'test_active')
    model.build_vocab(train, update=update)
    model.train(train, total_examples=model.corpus_count, epochs=5)
    end_time = datetime.datetime.now()
    print('Took {0} seconds.'.format((end_time - start_time).seconds))

model.save('avito_big_150m_sg0_size100.w2v')
