"""Load data and train model."""
import pickle
import lightgbm as lgb
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


class SingleModel(object):
    """Load data and train single model.

    Model can be trained on all data, on N-folds,
    can show cross-validation score and can be used to create meta-features.
    """

    def __init__(self):
        """Define parameters."""
        self.X_meta_active = np.load('data/X_meta_active_texts_sk.npy')
        self.X_test_meta_active = np.load('data/X_test_meta_active_texts_sk.npy')
        self.X = np.load('data/X_meta_from_texts.npy')
        self.X_test = np.load('data/X_meta_from_texts_test.npy')
        self.price_train = np.load('data/price_log_tree_usual_train.npy')
        self.price_test = np.load('data/price_log_tree_usual_test.npy')
        self.y = np.load('data/y.npy')
        self.kf = KFold(n_splits=5)
        with open('pickles/1_5_sublinear_tf-True_smooth_idf-False/X_texts.pickle', 'rb') as f:
            self.X_texts = pickle.load(f)
    
        with open('pickles/1_5_sublinear_tf-True_smooth_idf-False/X_test_texts.pickle', 'rb') as f:
            self.X_test_texts = pickle.load(f)

    def rmse(self, predictions, targets):
        """Metric."""
        return np.sqrt(((np.clip(predictions, 0, 1) - targets) ** 2).mean())

    def load_data(self, cat_features_name, price='tree', use_agg=False):
        """Load data."""
        if price == 'tree':
            pass
        elif price == 'non_tree':
            self.price_train = np.load('data/price_log_usual_train.npy')
            self.price__test = np.load('data/price_log_usual_test.npy')

        other_columns_usual_train = np.load('data/other_columns_usual_train.npy')
        other_columns_usual_test = np.load('data/other_columns_usual_test.npy')

        cat_features_train = np.load(cat_features_name)
        cat_features_test = np.load(cat_features_name.replace('train', 'test'))
        if use_agg:
            tr_agg = np.load('data/tr_agg.npy')
            te_agg = np.load('data/te_agg.npy')
        self.X_meta_ = np.hstack([self.X, self.X_meta_active, self.price_train.reshape(-1, 1),
                                  other_columns_usual_train, cat_features_train])
        self.X_test_full = np.hstack([self.X_test, self.X_test_meta_active, self.price_test.reshape(-1, 1),
                                      other_columns_usual_test, cat_features_test])

        if use_agg:
            self.X_meta_ = np.hstack([self.X, self.X_meta_active, self.price_train.reshape(-1, 1),
                                      other_columns_usual_train, cat_features_train, tr_agg])
            self.X_test_full = np.hstack([self.X_test, self.X_test_meta_active, self.price_test.reshape(-1, 1),
                                          other_columns_usual_test, cat_features_test, te_agg])

        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(self.X_meta_,
                                                                                  self.y.reshape(-1,),
                                                                                  test_size=0.20,
                                                                                  random_state=42)

    def train_model(self, model_type, params=None):
        """Train model."""
        if model_type == 'lgb':
            model = lgb.train(params, lgb.Dataset(self.X_train, label=self.y_train), 10000,
                              lgb.Dataset(self.X_valid, label=self.y_valid),
                              verbose_eval=100, early_stopping_rounds=100)

            print(self.rmse(model.predict(self.X_valid), self.y_valid))

    def create_meta_feature(self, feature_name, model):
        """Create meta-feature."""
        X_meta = np.zeros((1503424, 1))
        X_test_meta = []
        for fold_i, (train_i, test_i) in enumerate(self.kf.split(self.X_meta_)):
            print('Fold:', fold_i)
            clf = model
            clf.fit(self.X_meta_[train_i][:, 4:], self.y[train_i])
            X_meta[test_i, :] = clf.predict(self.X_meta_[test_i][:, 4:]).reshape(-1, 1)
            X_test_meta.append(clf.predict(self.X_test_full[:, 4:]))
        X_test_meta = np.stack(X_test_meta)
        X_test_meta_mean = np.mean(X_test_meta, axis=0)

        np.save('data/meta/' + feature_name + '_train.npy', X_meta)
        np.save('data/meta/' + feature_name + '_test.npy', X_test_meta_mean)

    def create_meta_feature_lgb(self, feature_name, params, text='only_text'):
        """Create meta-feature."""
        X_meta = np.zeros((1503424, 1))
        X_test_meta = []
        for fold_i, (train_i, test_i) in enumerate(self.kf.split(self.X_meta_)):
            print('Fold:', fold_i)
            if text == 'all':
                X_full = csr_matrix(hstack([self.X_texts, self.X_meta_[:, 4:].astype(float)]))
                X_test_full = csr_matrix(hstack([self.X_texts, self.X_meta_[:, 4:].astype(float)]))

                model = lgb.train(params, lgb.Dataset(X_full.tocsr()[train_i], label=self.y[train_i].reshape(-1,)), 10000,
                              lgb.Dataset(X_full.tocsr()[test_i], label=self.y[test_i].reshape(-1,)),
                              verbose_eval=100, early_stopping_rounds=100)
                X_meta[test_i, :] = model.predict(X_full.tocsr()[test_i]).reshape(-1, 1)
                X_test_meta.append(model.predict(X_test_full.tocsr()))

            elif text == 'only_text':
                model = lgb.train(params, lgb.Dataset(self.X_texts[train_i][:, 4:], label=self.y[train_i].reshape(-1,)), 10000,
                              lgb.Dataset(self.X_texts[test_i][:, 4:], label=self.y[test_i].reshape(-1,)),
                              verbose_eval=100, early_stopping_rounds=100)
                X_meta[test_i, :] = model.predict(self.X_texts[test_i][:, 4:]).reshape(-1, 1)
                X_test_meta.append(model.predict(self.X_test_texts[:, 4:]))

            else:
                model = lgb.train(params, lgb.Dataset(self.X_meta_[train_i][:, 4:], label=self.y[train_i].reshape(-1,)), 10000,
                              lgb.Dataset(self.X_meta_[test_i][:, 4:], label=self.y[test_i].reshape(-1,)),
                              verbose_eval=100, early_stopping_rounds=100)
                X_meta[test_i, :] = model.predict(self.X_meta_[test_i][:, 4:]).reshape(-1, 1)
                X_test_meta.append(model.predict(self.X_test_full[:, 4:]))

        X_test_meta = np.stack(X_test_meta)
        X_test_meta_mean = np.mean(X_test_meta, axis=0)

        np.save('data/meta/' + feature_name + '_train.npy', X_meta)
        np.save('data/meta/' + feature_name + '_test.npy', X_test_meta_mean)