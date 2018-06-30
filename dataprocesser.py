"""Feature generation."""
import os
import re
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
stop_words = set(stopwords.words('russian'))


class DataProcesser(object):
    """Process data for Avito competition."""

    def __init__(self, folder=''):
        """Init."""
        self.train = pd.read_csv(os.path.join(folder, 'train.csv.zip'), compression='zip')
        self.test = pd.read_csv(os.path.join(folder, 'test.csv.zip'), compression='zip')
        self.train_active = pd.read_csv(os.path.join(folder, 'train_active.csv.zip'), compression='zip')
        self.test_active = pd.read_csv(os.path.join(folder, 'test_active.csv.zip'), compression='zip')
        self.periods_train = pd.read_csv(os.path.join(folder, 'periods_train.csv.zip'), compression='zip')
        self.sub = pd.read_csv(os.path.join(folder, 'sample_submission.csv'))

    def save_to_pickle(self, object, name):
        """Save DF or Series to pickle."""
        with open('pickles/data/' + name + '.pickle', 'wb') as f:
            pickle.dump(object, f)

    def save_to_npy(self, object, name):
        """Save to npy."""
        np.save('data/' + name + '.npy', object)

    def target_encode(self, trn_series=None,
                      tst_series=None,
                      target=None,
                      min_samples_leaf=1,
                      smoothing=1,
                      noise_level=0):
        """
        Do mean encoding.

        https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features
        Smoothing is computed like in the following paper by Daniele Micci-Barreca
        https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
        trn_series : training categorical feature as a pd.Series
        tst_series : test categorical feature as a pd.Series
        target : target data as a pd.Series
        min_samples_leaf (int) : minimum samples to take category average into account
        smoothing (int) : smoothing effect to balance categorical average vs prior.
        """
        assert len(trn_series) == len(target)
        assert trn_series.name == tst_series.name
        temp = pd.concat([trn_series, target], axis=1)
        # Compute target mean
        averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
        # Compute smoothing
        smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
        # Apply average function to all target data
        prior = target.mean()
        # The bigger the count the less full_avg is taken into account
        averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
        averages.drop(["mean", "count"], axis=1, inplace=True)
        # Apply averages to trn and tst series
        ft_trn_series = pd.merge(
            trn_series.to_frame(trn_series.name),
            averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
            on=trn_series.name,
            how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
        # pd.merge does not keep the index so restore it
        ft_trn_series.index = trn_series.index
        ft_tst_series = pd.merge(
            tst_series.to_frame(tst_series.name),
            averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
            on=tst_series.name,
            how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
        # pd.merge does not keep the index so restore it
        ft_tst_series.index = tst_series.index
        return ft_trn_series, ft_tst_series

    def combine_dataset(self, dataset):
        """Make combined dataset."""
        if dataset == 'usual':
            df = pd.concat([self.train.drop('deal_probability', axis=1), self.test], axis=0)

        else:
            # folder = ''
            # self.train = pd.read_csv(os.path.join(folder, 'train.csv.zip'), compression='zip')
            # self.test = pd.read_csv(os.path.join(folder, 'test.csv.zip'), compression='zip')
            # self.train_active = pd.read_csv(os.path.join(folder, 'train_active.csv.zip'), compression='zip')
            # self.test_active = pd.read_csv(os.path.join(folder, 'test_active.csv.zip'), compression='zip')
            self.train_active['image'] = 0
            self.train_active['image_top_1'] = 0
            self.test_active['image'] = 0
            self.test_active['image_top_1'] = 0
            df = pd.concat([self.train.drop('deal_probability', axis=1), self.test, self.train_active, self.test_active], axis=0)
        df['duplicated_item_id'] = df.groupby(['user_id', 'description'])['price'].transform('count')
        df.drop_duplicates(['user_id', 'item_id', 'price'], inplace=True)
        return df

    def process_df(self, price_log=True, # category_encoding='mean',
                   categorical_features=["region", "city", "parent_category_name", "category_name", "user_type", "param_1", "param_2", "param_3", 'item_seq_number']):
        """
        Process df.

        Combine train and test data into one dataframe. Process it and create new features.
        - create featuren showing date and weekday; weather data has an image;
        - combine params into one;
        - possible to take log1p if price;
        - if model will be tree-based, price can be filled with an outlier value, otherwise it can be filled with median;
        - text-based features: length, count of words etc;
        - aggregates;
        - encoding of categorical variables with means or with LabelEncoder;


        """
        self.y = self.train['deal_probability']

        for dataset in ['usual', 'full']:
            print(dataset)
            df = self.combine_dataset(dataset)

            # need to reset index. Otherwise can't fillna with median
            df.reset_index(drop=True, inplace=True)

            df['activation_date'] = pd.to_datetime(df['activation_date'])
            df['date'] = df['activation_date'].dt.date
            df['weekday'] = df['activation_date'].dt.weekday

            # creating features showing that this sample has null value in this feature
            df['has_image'] = 1
            if dataset == 'usual':
                df.loc[df['image'].isnull(), 'has_image'] = 0
            else:
                df[:self.train.shape[0] + self.test.shape[0]].loc[df['image'].isnull(), 'has_image'] = 0
            df['price_is_null'] = 0
            df.loc[df['price'].isnull(), 'price_is_null'] = 1

            # features showing that there was some value
            df['param_1_is_null'] = 0
            df.loc[df['param_1'].isnull(), 'param_1_is_null'] = 1
            df['param_2_is_null'] = 0
            df.loc[df['param_2'].isnull(), 'param_2_is_null'] = 1
            df['param_3_is_null'] = 0
            df.loc[df['param_3'].isnull(), 'param_3_is_null'] = 1
            df['image_top_1_is_null'] = 0
            if dataset == 'usual':
                df.loc[df['image_top_1'].isnull(), 'image_top_1_is_null'] = 1
            else:
                df[:self.train.shape[0] + self.test.shape[0]].loc[df['image_top_1'].isnull(), 'image_top_1_is_null'] = 1

            df['params'] = df['param_1'].fillna('').astype(str) + ' ' + df['param_2'].fillna('').astype(str) + ' ' + df['param_3'].fillna('').astype(str)
            df['params'] = df['params'].str.strip()

            # At first I fillna with median values to use it for aggregating
            df['price'] = df.groupby(['city', 'category_name'])['price'].apply(lambda x: x.fillna(x.median()))
            df['price'] = df.groupby(['region', 'category_name'])['price'].apply(lambda x: x.fillna(x.median()))
            df['price'] = df.groupby(['category_name'])['price'].apply(lambda x: x.fillna(x.median()))

            # text based features.
            df['description'] = df['description'].apply(lambda x: str(x).replace('/\n', ' ').replace('\xa0', ' ').replace('.', '. ').replace(',', ', '))

            df['description'] = df['description'].fillna('')
            df['title'] = df['title'].fillna('')
            df['params'] = df['params'].fillna('')

            df['len_description'] = df['description'].apply(lambda x: len(x))
            df['words_description'] = df['description'].apply(lambda x: len(x.split()))
            df['average_description_word_length'] = df['len_description'] / df['words_description']

            df['len_title'] = df['title'].apply(lambda x: len(x))
            df['words_title'] = df['title'].apply(lambda x: len(x.split()))
            df['average_title_word_length'] = df['len_title'] / df['words_title']

            df['len_params'] = df['params'].apply(lambda x: len(x))
            df['words_params'] = df['params'].apply(lambda x: len(x.split()))
            df['average_params_word_length'] = df['len_params'] / df['words_params']

            df['title_to_params_length'] = df['len_title'] / df['len_params']
            df['title_to_params_words'] = df['words_title'] / df['words_params']

            print('generating text features')
            df['digit_count_description'] = df['description'].apply(lambda x: len(re.findall('[0-9]', str(x))))
            df['punctuation_count_description'] = df['description'].apply(lambda x: len(re.findall("[.,!?\"'():;-]", str(x))))
            df['symbols_count_description'] = df['description'].apply(lambda x: len(re.findall("[^0-9а-яёА-Яa-zA-Z \\n/.,!?\"'():;-]", str(x))))
            df['uppercase_symbol_count_description'] = df['description'].apply(lambda x: len(re.findall('[А-ЯA-Z]', str(x))))
            df['uppercase_symbol_count_description_to_len'] = df['uppercase_symbol_count_description'] / df['len_description']

            df['digit_count_title'] = df['title'].apply(lambda x: len(re.findall('[0-9]', str(x))))
            df['punctuation_count_title'] = df['title'].apply(lambda x: len(re.findall("[.,!?\"'():;-]", str(x))))
            df['symbols_count_title'] = df['title'].apply(lambda x: len(re.findall("[^0-9а-яёА-Яa-zA-Z \\n/.,!?\"'():;-]", str(x))))
            df['uppercase_symbol_count_title'] = df['title'].apply(lambda x: len(re.findall('[А-ЯA-Z]', str(x))))

            df['digit_count_params'] = df['params'].apply(lambda x: len(re.findall('[0-9]', str(x))))
            df['punctuation_count_params'] = df['params'].apply(lambda x: len(re.findall("[.,!?\"'():;-]", str(x))))
            df['symbols_count_params'] = df['params'].apply(lambda x: len(re.findall("[^0-9а-яёА-Яa-zA-Z \\n/.,!?\"'():;-]", str(x))))
            df['uppercase_symbol_count_params'] = df['params'].apply(lambda x: len(re.findall('[А-ЯA-Z]', str(x))))

            df['symbol1_count'] = df['description'].str.count('↓')
            df['symbol2_count'] = df['description'].str.count('\*')
            df['symbol3_count'] = df['description'].str.count('✔')
            df['symbol4_count'] = df['description'].str.count('❀')
            df['symbol5_count'] = df['description'].str.count('➚')
            df['symbol6_count'] = df['description'].str.count('ஜ')
            df['symbol7_count'] = df['description'].str.count('.')
            df['symbol8_count'] = df['description'].str.count('!')
            df['symbol9_count'] = df['description'].str.count('\?')
            df['symbol10_count'] = df['description'].str.count('  ')
            df['symbol11_count'] = df['description'].str.count('-')
            df['symbol12_count'] = df['description'].str.count(',')
            df['symbol13_count'] = df['description'].str.count('/')

            df['max_repeats_description'] = df['description'].apply(lambda x: Counter(x.lower().split()).most_common()[0][1])

            df['upper_case_word_count'] = df['description'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))
            df['stopword_count'] = df['description'].apply(lambda x: len([wrd for wrd in x.split() if wrd.lower() in stop_words]))

            # fill na
            if dataset == 'usual':
                df['image_top_1'].fillna(-1, inplace=True)
            else:
                df[:self.train.shape[0] + self.test.shape[0]]['image_top_1'].fillna(-1, inplace=True)
            df['param_1'].fillna(-1, inplace=True)
            df['param_2'].fillna(-1, inplace=True)
            df['param_3'].fillna(-1, inplace=True)

            if price_log:
                df['price'] = np.log1p(df['price'])

            # aggregates
            print('generating aggregates')
            df['user_price_mean'] = df.groupby('user_id')['price'].transform('mean')
            df['user_price_median'] = df.groupby('user_id')['price'].transform('median')
            df['user_ad_sum'] = df.groupby('user_id')['price'].transform('sum')

            df['region_price_mean'] = df.groupby('region')['price'].transform('mean')
            df['region_price_median'] = df.groupby('region')['price'].transform('median')
            df['region_price_max'] = df.groupby('region')['price'].transform('max')
            df['region_price_sum'] = df.groupby('region')['price'].transform('sum')

            df['item_seq_number_price_mean'] = df.groupby('item_seq_number')['price'].transform('mean')
            df['item_seq_number_price_median'] = df.groupby('item_seq_number')['price'].transform('median')
            df['item_seq_number_price_max'] = df.groupby('item_seq_number')['price'].transform('max')
            df['item_seq_number_price_sum'] = df.groupby('item_seq_number')['price'].transform('sum')

            df['city_price_mean'] = df.groupby('city')['price'].transform('mean')
            df['city_price_median'] = df.groupby('city')['price'].transform('median')
            df['city_price_max'] = df.groupby('city')['price'].transform('max')
            df['city_price_sum'] = df.groupby('city')['price'].transform('sum')

            df['day_ads_sum'] = df.groupby('date')['price'].transform('count')

            df['parent_category_name_price_mean'] = df.groupby('parent_category_name')['price'].transform('mean')
            df['parent_category_name_price_median'] = df.groupby('parent_category_name')['price'].transform('median')
            df['parent_category_name_price_max'] = df.groupby('parent_category_name')['price'].transform('max')
            df['parent_category_name_price_sum'] = df.groupby('parent_category_name')['price'].transform('sum')

            df['category_name_price_mean'] = df.groupby('category_name')['price'].transform('mean')
            df['category_name_price_median'] = df.groupby('category_name')['price'].transform('median')
            df['category_name_price_max'] = df.groupby('category_name')['price'].transform('max')
            df['category_name_price_sum'] = df.groupby('category_name')['price'].transform('sum')

            df['user_type_category_price_mean'] = df.groupby(['user_type', 'parent_category_name'])['price'].transform('mean')
            df['user_type_category_price_median'] = df.groupby(['user_type', 'parent_category_name'])['price'].transform('median')
            df['user_type_category_price_max'] = df.groupby(['user_type', 'parent_category_name'])['price'].transform('max')
            df['user_type_category_price_sum'] = df.groupby(['user_type', 'parent_category_name'])['price'].transform('sum')

            df['region_price_mean'] = df.groupby('region')['price'].transform('mean')
            df['region_price_median'] = df.groupby('region')['price'].transform('median')
            df['region_price_max'] = df.groupby('region')['price'].transform('max')
            df['region_price_sum'] = df.groupby('region')['price'].transform('sum')

            df['user_type_price_mean'] = df.groupby('user_type')['price'].transform('mean')
            df['user_type_price_median'] = df.groupby('user_type')['price'].transform('median')
            df['user_type_price_max'] = df.groupby('user_type')['price'].transform('max')
            df['user_type_price_sum'] = df.groupby('user_type')['price'].transform('sum')
            
            df['param_1_price_mean'] = df.groupby('param_1')['price'].transform('mean')
            df['param_1_price_median'] = df.groupby('param_1')['price'].transform('median')
            df['param_1_price_max'] = df.groupby('param_1')['price'].transform('max')
            df['param_1_price_sum'] = df.groupby('param_1')['price'].transform('sum')

            df['param_2_price_mean'] = df.groupby('param_2')['price'].transform('mean')
            df['param_2_price_median'] = df.groupby('param_2')['price'].transform('median')
            df['param_2_price_max'] = df.groupby('param_2')['price'].transform('max')
            df['param_2_price_sum'] = df.groupby('param_2')['price'].transform('sum')

            df['param_3_price_mean'] = df.groupby('param_3')['price'].transform('mean')
            df['param_3_price_median'] = df.groupby('param_3')['price'].transform('median')
            df['param_3_price_max'] = df.groupby('param_3')['price'].transform('max')
            df['param_3_price_sum'] = df.groupby('param_3')['price'].transform('sum')
            
            df['user_type_region_price_mean'] = df.groupby(['user_type', 'region'])['price'].transform('mean')
            df['user_type_region_price_median'] = df.groupby(['user_type', 'region'])['price'].transform('median')
            df['user_type_region_price_max'] = df.groupby(['user_type', 'region'])['price'].transform('max')
            df['user_type_region_price_sum'] = df.groupby(['user_type', 'region'])['price'].transform('sum')

            df['category_name_param_1_price_mean'] = df.groupby(['category_name', 'param_1'])['price'].transform('mean')
            df['category_name_param_2_price_mean'] = df.groupby(['category_name', 'param_2'])['price'].transform('mean')
            df['category_name_param_3_price_mean'] = df.groupby(['category_name', 'param_3'])['price'].transform('mean')

            if dataset == 'usual':
                df['category_name_image_top_1_price_mean'] = df.groupby(['category_name', 'image_top_1'])['price'].transform('mean')
                df['city_category_name_image_top_1_price_mean'] = df.groupby(['city', 'category_name', 'image_top_1'])['price'].transform('mean')

                df['image_top_1_price_mean'] = df.groupby('image_top_1')['price'].transform('mean')
                df['image_top_1_price_median'] = df.groupby('image_top_1')['price'].transform('median')
                df['image_top_1_price_max'] = df.groupby('image_top_1')['price'].transform('max')
                df['image_top_1_price_sum'] = df.groupby('image_top_1')['price'].transform('sum')

                df['user_id_image_top_1_price_sum'] = df.groupby(['user_id', 'image_top_1'])['price'].transform('sum')

            else:
                df['category_name_image_top_1_price_mean'] = 0
                df['city_category_name_image_top_1_price_mean'] = 0
                df['image_top_1_price_mean'] = 0
                df['image_top_1_price_median'] = 0
                df['image_top_1_price_max'] = 0
                df['image_top_1_price_sum'] = 0
                df['user_id_image_top_1_price_sum'] = 0

                df[:self.train.shape[0] + self.test.shape[0]]['category_name_image_top_1_price_mean'] = df[:self.train.shape[0] + self.test.shape[0]].groupby(['category_name', 'image_top_1'])['price'].transform('mean')
                df[:self.train.shape[0] + self.test.shape[0]]['city_category_name_image_top_1_price_mean'] = df[:self.train.shape[0] + self.test.shape[0]].groupby(['city', 'category_name', 'image_top_1'])['price'].transform('mean')

                df[:self.train.shape[0] + self.test.shape[0]]['image_top_1_price_mean'] = df[:self.train.shape[0] + self.test.shape[0]].groupby('image_top_1')['price'].transform('mean')
                df[:self.train.shape[0] + self.test.shape[0]]['image_top_1_price_median'] = df[:self.train.shape[0] + self.test.shape[0]].groupby('image_top_1')['price'].transform('median')
                df[:self.train.shape[0] + self.test.shape[0]]['image_top_1_price_max'] = df[:self.train.shape[0] + self.test.shape[0]].groupby('image_top_1')['price'].transform('max')
                df[:self.train.shape[0] + self.test.shape[0]]['image_top_1_price_sum'] = df[:self.train.shape[0] + self.test.shape[0]].groupby('image_top_1')['price'].transform('sum')

                df[:self.train.shape[0] + self.test.shape[0]]['user_id_image_top_1_price_sum'] = df[:self.train.shape[0] + self.test.shape[0]].groupby(['user_id', 'image_top_1'])['price'].transform('sum')

            df['user_id_category_name_price_sum'] = df.groupby(['user_id', 'category_name'])['price'].transform('sum')
            
            df['user_type_region_price_sum_to_user_type_sum'] = df['user_type_region_price_sum'] / df['user_type_price_sum']

            df['user_id_category_name_price_sum_to_user_id_sum'] = df['user_id_category_name_price_sum'] / df.groupby(['user_id'])['price'].transform('sum')
            df['user_id_category_name_price_sum_to_category_name_sum'] = df['user_id_category_name_price_sum'] / df.groupby(['category_name'])['price'].transform('sum')

            df['user_len_description_mean'] = df.groupby('user_id')['len_description'].transform('mean')
            df['user_len_description_sum'] = df.groupby('user_id')['len_description'].transform('sum')

            df['user_len_title_mean'] = df.groupby('user_id')['len_title'].transform('mean')
            df['user_len_title_sum'] = df.groupby('user_id')['len_title'].transform('sum')
            
            df['average_params_word_length'].fillna(0, inplace=True)
            df.drop(['date', 'user_id', 'activation_date', 'title', 'params', 'description'], axis=1, inplace=True)
            if dataset == 'usual':
                df.drop(['image'], axis=1, inplace=True)

            for i in ['tree', 'non_tree']:
                print(i)
                if i == 'tree':
                    df1 = df.copy()
                    df1.loc[df1['price_is_null'] == 1, 'price'] = -999
                    df1.fillna(-999, inplace=True)
                    self.train = df1[:self.train.shape[0]]
                    self.test = df1[self.train.shape[0]:self.train.shape[0] + self.test.shape[0]]
                    self.save_to_npy(self.train['price'], 'price_log_tree_' + dataset + '_train')
                    self.save_to_npy(self.test['price'], 'price_log_tree_' + dataset + '_test')
                    if dataset == 'full':
                        self.train_active = df[self.train.shape[0] + self.test.shape[0]:self.train.shape[0] + self.test.shape[0] + self.train_active[0]]
                        self.test_active = df[self.train.shape[0] + self.test.shape[0] + self.train_active[0]:]
                        self.save_to_npy(self.train_active['price'], 'price_log_tree_train_active')
                        self.save_to_npy(self.test_active['price'], 'price_log_tree_test_active')
                    del df1
                else:
                    print('copying')
                    df1 = df.copy()
                    print('filling mean')
                    for col in df1.columns:
                        if df1[col].isnull().any():
                            print(col)
                            if df1[col].dtype == 'object':
                                df1[col] = df1[col].fillna('')
                            else:
                                df1[col] = df1[col].fillna(df1[col].mean())
                    print('getting data')
                    self.train = df1[:self.train.shape[0]]
                    self.test = df1[self.train.shape[0]:self.train.shape[0] + self.test.shape[0]]
                    print('saving data')
                    self.save_to_npy(self.train['price'], 'price_log_' + dataset + '_train')
                    self.save_to_npy(self.test['price'], 'price_log_' + dataset + '_test')
                    if dataset == 'full':
                        print('getting active data')
                        self.train_active = df[self.train.shape[0] + self.test.shape[0]:self.train.shape[0] + self.test.shape[0] + self.train_active[0]]
                        self.test_active = df[self.train.shape[0] + self.test.shape[0] + self.train_active[0]:]
                        print('saving active data')
                        self.save_to_npy(self.train_active['price'], 'price_log_train_active')
                        self.save_to_npy(self.test_active['price'], 'price_log_test_active')
                    del df1

            for category_encoding in ['mean', 'non_mean', 'expanding']:
                print(category_encoding)
                if category_encoding == 'non_mean':
                    df1 = df.copy()
                    for col in categorical_features:
                        lbl = LabelEncoder()
                        df1[col] = lbl.fit_transform(df1[col].values.astype('str'))
                    self.train = df1[:self.train.shape[0]]
                    self.test = df1[self.train.shape[0]:self.train.shape[0] + self.test.shape[0]]

                    self.save_to_npy(self.train[categorical_features], 'cat_features_le_' + dataset + '_train')
                    self.save_to_npy(self.test[categorical_features], 'cat_features_le_' + dataset + '_test')

                    if dataset == 'full':
                        self.train_active = df[self.train.shape[0] + self.test.shape[0]:self.train.shape[0] + self.test.shape[0] + self.train_active[0]]
                        self.test_active = df[self.train.shape[0] + self.test.shape[0] + self.train_active[0]:]
                        self.save_to_npy(self.train_active[categorical_features], 'cat_features_le_train_active')
                        self.save_to_npy(self.test_active[categorical_features], 'cat_features_le_test_active')

                    del df1

                elif category_encoding == 'mean':
                    for min_samples_leaf in [1, 10, 50, 100]:
                        for smoothing in [1, 5, 10, 20]:
                            for noise_level in [0.0, 0.1, 0.01]:
                                print('min_samples_leaf', min_samples_leaf, 'smoothing', smoothing, 'noise_level', noise_level)
                                df1 = df.copy()
                                self.train = df1[:self.train.shape[0]]
                                self.test = df1[self.train.shape[0]:self.train.shape[0] + self.test.shape[0]]

                                for col in categorical_features:
                                    self.train[col], self.test[col] = self.target_encode(self.train[col], self.test[col], self.y,
                                        min_samples_leaf=min_samples_leaf, smoothing=smoothing, noise_level=noise_level)

                                self.save_to_npy(self.train[categorical_features], 'cat_features_me_' + dataset + '_train' + str(min_samples_leaf) + '_' + str(smoothing) + '_' + str(noise_level))
                                self.save_to_npy(self.test[categorical_features], 'cat_features_me_' + dataset + '_test' + str(min_samples_leaf) + '_' + str(smoothing) + '_' + str(noise_level))

                                if dataset == 'full':
                                    self.train_active = df[self.train.shape[0] + self.test.shape[0]:self.train.shape[0] + self.test.shape[0] + self.train_active[0]]
                                    self.test_active = df[self.train.shape[0] + self.test.shape[0] + self.train_active[0]:]
                                    self.save_to_npy(self.train_active[categorical_features], 'cat_features_me_train_active' + str(min_samples_leaf) + '_' + str(smoothing) + '_' + str(noise_level))
                                    self.save_to_npy(self.test_active[categorical_features], 'cat_features_me_test_active' + str(min_samples_leaf) + '_' + str(smoothing) + '_' + str(noise_level))
                                del df1

                else:
                    self.train = df[:self.train.shape[0]]
                    self.test = df[self.train.shape[0]:self.train.shape[0] + self.test.shape[0]]
                    for col in categorical_features:
                        
                        train = pd.concat([self.train, self.y], axis=0)
                        cum_sum = train.groupby(col)['deal_probability'].cumsum() - train['deal_probability']
                        cum_count = train.groupby(col).cumcount()

                        train[col + 'exp_enc'] = cum_sum / cum_count
                        train[col + 'exp_enc'].fillna(train['deal_probability'].mean(), inplace=True)
                        mapping = train.groupby(col)[col + 'exp_enc'].last().to_dict()
                        self.train[col + 'exp_enc'] = train[col + 'exp_enc']
                        self.test[col + 'exp_enc'] = self.test[col].apply(lambda x: mapping[x])
                        self.train.drop(col, axis=1, inplace=True)
                        self.test.drop(col, axis=1, inplace=True)

                    self.save_to_npy(self.train[categorical_features], 'cat_features_exp_enc_' + dataset + '_train')
                    self.save_to_npy(self.test[categorical_features], 'cat_features_exp_enc_' + dataset + '_test')

                    if dataset == 'full':
                        self.train_active = df[self.train.shape[0] + self.test.shape[0]:self.train.shape[0] + self.test.shape[0] + self.train_active[0]]
                        self.test_active = df[self.train.shape[0] + self.test.shape[0] + self.train_active[0]:]
                        self.save_to_npy(self.train_active[categorical_features], 'cat_features_exp_enc__train_active')
                        self.save_to_npy(self.test_active[categorical_features], 'cat_features_exp_enc__test_active')


            self.train = df[:self.train.shape[0]]
            self.test = df[self.train.shape[0]:self.train.shape[0] + self.test.shape[0]]

            cols_to_save = [i for i in df.columns if i not in categorical_features and i != 'price']
            self.save_to_npy(self.train[cols_to_save], 'other_columns_' + dataset + '_train')
            self.save_to_npy(self.test[cols_to_save], 'other_columns_' + dataset + '_test')

            if dataset == 'full':
                self.train_active = df[self.train.shape[0] + self.test.shape[0]:self.train.shape[0] + self.test.shape[0] + self.train_active[0]]
                self.test_active = df[self.train.shape[0] + self.test.shape[0] + self.train_active[0]:]
                self.save_to_npy(self.train_active[cols_to_save], 'other_columns_train_active')
                self.save_to_npy(self.test_active[cols_to_save], 'other_columns_test_active')