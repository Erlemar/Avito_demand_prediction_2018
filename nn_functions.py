"""Neural net helper functions."""

import pandas as pd
from keras.preprocessing import text, sequence
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping
import keras.backend as K
from sklearn import metrics
from sklearn.model_selection import train_test_split
import os

from gensim.models import FastText

import time 
import gc
import pickle

np.random.seed(42)

from keras.layers import Input, Dropout, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, Flatten, CuDNNGRU
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras import backend as K
from keras.models import Model

from sklearn.preprocessing import LabelBinarizer, LabelEncoder, MinMaxScaler, StandardScaler

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['OMP_NUM_THREADS'] = '6'

import threading
import multiprocessing
from multiprocessing import Pool, cpu_count
from contextlib import closing
cores = 6


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=0))

def tokenize(max_features, max_len, on='train', train_path='f:/avito/train.csv', test_path=None,
             tokenizer=None, clean_text=False, return_tokenizer=False, return_full_train=False):
    """
    Tokenize text.

    Read train and test data, process description feature, tokenize it.
    Parameters:
    - on: fit tokenizer on train or train + test;
    - train_path: path to train file;
    - test_path: past to test file;
    - max_features: tokenizer parameter;
    - max_len: tokenizer parameter;
    - tokenizer: can pass tokenizer with different parameters or use a default one;
    - clean_text: apply text cleaning or not;
    """
    # check that "on" has a correct value.
    assert on in ['train', 'all']

    print('Reading train data.')
    train = pd.read_csv(train_path, index_col=0)
    labels = train['deal_probability'].values
    train = train['description'].astype(str).fillna('')
    text = train

    # define tokenizer
    if tokenizer:
        tokenizer = tokenizer
    else:
        tokenizer = Tokenizer(num_words=max_features)

    if on == 'all':
        print('Reading test data.')
        test = pd.read_csv(test_path, index_col = 0)
        test = test['description'].astype(str).fillna('')
        text = text.append(test)

    # clean text
    if clean_text:
        pass
        # print('Cleaning.')

    print('Fitting.')
    tokenizer.fit_on_texts(text)

    # split data
    X_train, X_valid, y_train, y_valid = train_test_split(train,
                                                          labels,
                                                          test_size = 0.1,
                                                          random_state = 23)
    print('Converting to sequences.')
    X_train = tokenizer.texts_to_sequences(X_train)
    X_valid = tokenizer.texts_to_sequences(X_valid)
    if test_path:
        test = tokenizer.texts_to_sequences(test)

    print('Padding.')
    X_train = sequence.pad_sequences(X_train, maxlen=max_len)
    X_valid = sequence.pad_sequences(X_valid, maxlen=max_len)
    if test_path:
        test = sequence.pad_sequences(test, maxlen=max_len)

    data = {}
    data['X_train'] = X_train
    data['X_valid'] = X_valid
    data['y_train'] = y_train
    data['y_valid'] = y_valid
    if test_path:
        data['test'] = test

    if return_tokenizer:
        data['tokenizer'] = tokenizer

    if return_full_train:
        X = np.concatenate([X_train, X_valid])
        y = np.concatenate([y_train, y_valid])
        data['X'] = X
        data['y'] = y

    return data

def load_emb(embedding_path, tokenizer, max_features, default=False, embed_size=300):
    """Load embeddings."""

    fasttext_model = FastText.load(embedding_path)
    word_index = tokenizer.word_index

    # my pretrained embeddings have different index, so need to add offset.
    if default:
        nb_words = min(max_features, len(word_index))
    else:
        nb_words = min(max_features, len(word_index)) + 2

    embedding_matrix = np.zeros((nb_words, embed_size))

    for word, i in word_index.items():
        if i >= max_features: continue
        try:
            embedding_vector = fasttext_model[word]
        except KeyError:
            embedding_vector = None
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

def load_processed_data(return_all_train=False):
    """Load preprocessed data."""

    X_train = np.load('data/X_train_padded.npy')
    X_valid = np.load('data/X_valid_padded.npy')
    X_test = np.load('data/X_test_padded.npy')
    y_train = np.load('data/y_train_nn.npy')
    y_valid = np.load('data/y_valid_nn.npy')

    X = np.load('data/X_padded.npy')
    y = np.load('data/y_padded.npy')

    with open('keras_tokenizer.pickle', 'rb') as f:
        tokenizer = pickle.load(f)

    if return_all_train:
        return X, X_valid, X_test, y, y_valid, tokenizer
    else:
        return X_train, X_valid, X_test, y_train, y_valid, tokenizer



def get_keras_fasttext(features=[], feature_names=[]):
    """Convert arrays into keras input format."""
    data = {}
    for v, k in zip(features, feature_names):
        data[k] = v
    return data


def calc_decay(lr_init, lr_fin, train_size, batch_size, epochs, val_rate=None):
    """Calculate decay rate for Adam."""

    steps = (int(train_size / batch_size)) * epochs
    if val_rate:
        steps *= (1 - val_rate)
    exp_decay = lambda init, fin, steps: (init / fin) ** (1 / (steps - 1)) - 1
    lr_decay = exp_decay(lr_init, lr_fin, steps)

    return lr_decay