from dateutil.relativedelta import relativedelta
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import time

import keras.backend as K
import tensorflow as tf
from keras.layers import Activation, Dense, Dropout, LSTM
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential, model_from_json
from keras.optimizers import Adam
from scipy.signal import periodogram
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.linear_model import Ridge
from xgboost.sklearn import XGBRegressor

class TSEnsemble:
    def __init__(self, bags=1, diff=0, epochs=15, seed=42, verbose=0):
        # Raw original series
        self.series = None
        self.forecast = None
        
        # Splitted and transformed series
        self.dtrain = None
        self.dmeta = None
        self.dtest = None
        self.dfc = None

        # Execution params
        self.epochs = epochs
        self.verbose = verbose
        self.bags = bags
        self.seed = seed
        self.losses = {}
        self.scores = {
            'rmse': {},
            'mape': {}
        }
        
        # Intermediate holders
        self.meta_true = []
        self.meta_pred = {}
        self.test_true = []
        self.test_pred = {}
        self.fc_pred = {}

        # Transformation values
        self.diff = diff
        self.mean = None
        self.std = None
        
        # Network values
        self.networks = 1
        self.num_fc = None
        self.max_sl = None
        
        # Model base and meta params
        self.base_params = {
            'sl': [], # Sequence Length
            'hl': [2], # Hidden Layers
            'lr': [.01] # Learning Rate
        }
        self.meta_params = {
            'rf': { # RandomForest
                'n_estimators': 100
            },
            'ridge': { # Ridge
                'alpha': 1.
            },
            'xgb': { # XGB
                'n_estimators': 100,
                'learning_rate': .1,
                'max_depth': 3
            }
        }

    def set_params(self, base_params, meta_params):
        for key, val in base_params.items():
            self.base_params[key] = val

        sls = self.base_params['sl']
        if type(sls) == dict:
            self.base_params['sl'] = self.get_fourier_lags(self.series, n=sls['n'], min=sls['min'], max=sls['max'])
        elif len(sls) == 0:
            self.base_params['sl'] = self.get_fourier_lags(self.series)

        self.num_fc = min(self.base_params['sl'])
        self.max_sl = min(self.base_params['sl']) + max(self.base_params['sl'])

        self.networks = 1
        for key, val in self.base_params.items():
            self.networks *= len(val)

        for key, val in meta_params.items():
            for key2, val2 in val.items():
                self.meta_params[key][key2] = val2

        return self.networks, self.num_fc, self.max_sl

    def init_series(self, data, hold_split=.2, test_split=.2):
        self.series = data.copy()

        if self.diff > 0:
            data = self.diff_transform(data)

        hold_split_len = round(len(data) * hold_split)

        train = data[:-hold_split_len]
        hold = data[-hold_split_len:]

        test_split_len = round(len(hold) * test_split)

        meta = hold[:-test_split_len]
        test = hold[-test_split_len:]

        self.dtrain = self.norm_transform(train)
        self.dmeta = self.norm_transform(meta)
        self.dtest = self.norm_transform(test)

        return self.dtrain, self.dmeta, self.dtest

    def init_forecast(self, data=None):
        if data is None:
            data = self.series[-(self.max_sl + self.diff):]

        self.forecast = data.copy()

        if self.diff > 0:
            data = self.diff_transform(data)
        
        self.dfc = self.norm_transform(data)

        return self.dfc

    def diff_transform(self, data):
        data = data - data.shift(self.diff)

        return data[self.diff:].dropna()

    def diff_reverse(self, data, window):
        data = window[:self.diff].add(data, fill_value=0)

        for i, val in enumerate(data):
            if i >= self.diff:
                data[i] = val + data[i - self.diff]

        return data[self.diff:].dropna()

    def norm_transform(self, data):
        if self.mean is None or self.std is None:
            self.mean = data.mean()
            self.std = data.std()
        
        return (data - self.mean) / self.std

    def norm_reverse(self, data):
        return (data * self.std) + self.mean

    def get_fourier_lags(self, data, n=5, min=1, max=1000):
        frec, spec = periodogram(data)
        
        df = pd.DataFrame({'freq': frec, 'spec': spec}).sort_values(by='spec', ascending=False)
        top_freqs = df.freq.values
        top_lags = np.array([round(1 / i) for i in top_freqs if i != 0])
        _, idx = np.unique(top_lags, return_index=True)
        
        lags = top_lags[np.sort(idx)]
        lags = np.int_(lags)
        lags = lags[lags >= min]
        lags = lags[lags <= max]
        lags = lags[:n]
        
        if len(lags) < n:
            print('Warning: Too little Fourier sequences available')
        
        return lags

    def get_rmse(self, true, pred):
        return np.sqrt(np.mean((true - pred) ** 2))

    def get_mape(self, true, pred):
        return np.mean(np.abs((true - pred) / true)) * 100

    def get_mean_corr(self, preds=None):
        if preds is None:
            preds = self.meta_pred

        preds = np.array(list(preds.values()))
        preds = preds.reshape(preds.shape[0], (preds.shape[1] * preds.shape[2]))
        
        corrs = []

        for i in range(len(preds)):
            for j in range(len(preds)):
                if i < j:
                    r, p = pearsonr(preds[i], preds[j])
                    corrs.append(r)

        return np.mean(corrs)

    def series_to_seq(self, series, sl):
        seq = []

        for i in range(len(series) - sl + 1):
            seq.append(series[i:(i + sl)].values)

        seq = np.array(seq)

        return seq

    def series_to_fc(self, series, sl):
        fc = series[-sl:].values

        return fc

    def seq_to_series(self, y_pred, y_true=None):
        if not y_true:
            y_true = self.test_true
        
        s_pred = []
        s_true = []

        n_seq = y_pred.shape[0]
        idx_from = self.max_sl - self.num_fc
        for i in range(0, n_seq):
            idx = self.dtest[(idx_from + i):(idx_from + i + self.num_fc)].index
            window = self.series[(idx[0] - relativedelta(days=self.diff)):][:self.diff]
            
            t = pd.Series(y_true[i], index=idx)
            t = self.norm_reverse(t)
            t = self.diff_reverse(t, window)
            s_true.append(t)

            p = pd.Series(y_pred[i], index=idx)
            p = self.norm_reverse(p)
            p = self.diff_reverse(p, window)
            s_pred.append(p)

        return s_pred, s_true

    def fc_to_series(self, y_fc):
        idx = np.array([self.dfc.index[-1] + relativedelta(days=(i + 1)) for i in range(0, self.num_fc)])
        window = self.forecast[(self.dfc.index[-1] - relativedelta(days=(self.diff - 1))):][:self.diff]

        s_fc = pd.Series(y_fc, index=idx)
        s_fc = self.norm_reverse(s_fc)
        s_fc = self.diff_reverse(s_fc, window)

        return s_fc

    def get_scores(self, y_pred, y_true=None):
        s_pred, s_true = self.seq_to_series(y_pred, y_true)
        scores = {'rmse': .0, 'mape': .0}

        n_seq = len(s_pred)
        for i in range(0, n_seq):
            scores['rmse'] += self.get_rmse(s_true[i], s_pred[i])
            scores['mape'] += self.get_mape(s_true[i], s_pred[i])

        scores['rmse'] /= float(n_seq)
        scores['mape'] /= float(n_seq)

        return scores

    def get_models(self):
        names = list(self.scores['rmse'].keys())

        return names

    def best_model(self):
        name = None
        scores = {'rmse': np.Inf, 'mape': np.Inf}

        for key, val in self.scores['rmse'].items():
            if val < scores['rmse']:
                name = key
                scores['rmse'] = val
                scores['mape'] = self.scores['mape'][key]

        return name, scores

    def best_pred(self, idx=None):
        name, _ = self.best_model()

        y_fc = self.test_pred[name]
        s_fc, _ = self.seq_to_series(y_fc)

        if idx:
            s_fc = s_fc[idx]

        return s_fc

    def best_forecast(self):
        name, _ = self.best_model()

        y_fc = self.fc_pred[name]
        s_fc = self.fc_to_series(y_fc)

        return s_fc

    def get_meta_train(self):
        x_train = np.array(list(self.meta_pred.values()))
        x_train = x_train.reshape(x_train.shape[0], (x_train.shape[1] * x_train.shape[2])).transpose()

        y_train = np.array(self.meta_true)
        y_train = y_train.reshape(y_train.shape[0] * y_train.shape[1])

        return x_train, y_train

    def get_meta_test(self):
        x_test = np.array(list(self.test_pred.values()))[:self.networks]
        x_test = x_test.reshape(x_test.shape[0], (x_test.shape[1] * x_test.shape[2])).transpose()

        return x_test

    def get_meta_forecast(self):
        x_fc = np.array(list(self.fc_pred.values()))[:self.networks].transpose()

        return x_fc

    def base_build_model(self, nn, hl, lr):
        model = Sequential()

        model.add(LSTM(nn, return_sequences=True, input_shape=(None, 1)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(.2))

        for i in range(hl):
            model.add(LSTM(nn, return_sequences=True))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(.2))

        model.add(TimeDistributed(Dense(1)))
        model.add(Activation('linear'))

        model.compile(loss='mse', optimizer=Adam(lr=lr), sample_weight_mode='temporal')

        return model

    def meta_build_model(self, key, seed):
        model = None

        if key == 'rf':
            model = RandomForestRegressor(
                n_estimators = self.meta_params['rf']['n_estimators'],
                verbose = 1 if self.verbose > 1 else 0,
                random_state = seed,
                n_jobs = -1
            )
        elif key == 'ridge':
            model = Ridge(
                alpha = self.meta_params['ridge']['alpha'],
                random_state = seed
            )
        elif key == 'xgb':
            model = XGBRegressor(
                n_estimators = self.meta_params['xgb']['n_estimators'],
                max_depth = self.meta_params['xgb']['max_depth'],
                learning_rate = self.meta_params['xgb']['learning_rate'],
                subsample = 0.8,
                colsample_bytree = 0.8,
                silent = False if self.verbose > 1 else True,
                random_state = seed,
                n_jobs = -1
            )

        return model

    def model_train(self, data=None, hold_split=.2, test_split=.2):
        if data:
            self.init_series(data, hold_split, test_split)

        # Fit RNNs
        for sl in self.base_params['sl']:
            for hl in self.base_params['hl']:
                for lr in self.base_params['lr']:
                    name = self.rnn_name(sl, hl, lr)
                    if self.verbose > 0:
                        print('Fitting {}...'.format(name))
                    self.losses[name] = self.base_train(self.dtrain, sl, hl, lr)

        # Predict RNNs meta data
        for sl in self.base_params['sl']:
            for hl in self.base_params['hl']:
                for lr in self.base_params['lr']:
                    name = self.rnn_name(sl, hl, lr)
                    if self.verbose > 0:
                        print('Processing {}...'.format(name))
                    self.meta_pred[name], self.meta_true = self.base_test(self.dmeta, sl, hl, lr)

        # Fit Metas
        for key in list(self.meta_params.keys()):
            name = self.meta_name(key)
            if self.verbose > 0:
                print('Fitting {}...'.format(name))
            self.meta_train(key)

    def model_test(self):
        # Test RNNs
        for sl in self.base_params['sl']:
            for hl in self.base_params['hl']:
                for lr in self.base_params['lr']:
                    name = self.rnn_name(sl, hl, lr)
                    if self.verbose > 0:
                        print('Testing {}...'.format(name))
                    self.test_pred[name], self.test_true = self.base_test(self.dtest, sl, hl, lr)
                    scores = self.get_scores(self.test_pred[name])
                    for i in self.scores.keys():
                        self.scores[i][name] = scores[i]

        # Test RNNs Mean
        name = 'base_mean'
        if self.verbose > 0:
            print('Testing {}...'.format(name))
        self.test_pred[name] = self.mean_test()
        scores = self.get_scores(self.test_pred[name])
        for i in self.scores.keys():
            self.scores[i][name] = scores[i]

        # Test Metas
        for key in list(self.meta_params.keys()):
            name = self.meta_name(key)
            if self.verbose > 0:
                print('Testing {}...'.format(name))
            self.test_pred[name] = self.meta_test(key)
            scores = self.get_scores(self.test_pred[name])
            for i in self.scores.keys():
                self.scores[i][name] = scores[i]

    def model_forecast(self, data=None):
        self.init_forecast(data)

        # Forecast RNNs for meta learners
        for sl in self.base_params['sl']:
            for hl in self.base_params['hl']:
                for lr in self.base_params['lr']:
                    name = self.rnn_name(sl, hl, lr)
                    if self.verbose > 0:
                        print('Forecasting {}...'.format(name))
                    self.fc_pred[name] = self.base_forecast(self.dfc, sl, hl, lr)

        # Forecast RNNs Mean
        name = 'base_mean'
        if self.verbose > 0:
            print('Forecasting {}...'.format(name))
        self.fc_pred[name] = self.mean_forecast()

        # Forecast Metas
        for key in list(self.meta_params.keys()):
            name = self.meta_name(key)
            if self.verbose > 0:
                print('Forecasting {}...'.format(name))
            self.fc_pred[name] = self.meta_forecast(key)

    def base_train(self, data, sl, hl, lr):
        num_fc = sl
        nn = sl

        seq = self.series_to_seq(data, (sl + num_fc))
        loss = np.zeros(self.epochs)

        for i in range(0, self.bags):
            name = self.rnn_name(sl, hl, lr, i)
            if self.rnn_is_model(name):
                history = self.rnn_load_history(name)
                loss += np.array(history['loss'])
                continue

            seed = self.seed + i
            np.random.seed(seed)
            K.clear_session()
            tf.set_random_seed(seed)

            np.random.shuffle(seq)
            x_train = seq[:, :sl].reshape(seq.shape[0], sl, 1)
            y_train = seq[:, -num_fc:].reshape(seq.shape[0], num_fc, 1)

            model = self.base_build_model(nn, hl, lr)
            h = model.fit(
                x_train, y_train,
                batch_size = 32,
                epochs = self.epochs,
                validation_split = .1,
                verbose = 1 if self.verbose > 1 else 0
            )

            history = h.history
            loss += np.array(history['loss'])

            self.rnn_save_model(model, name)
            self.rnn_save_history(history, name)

        loss = list(loss / float(self.bags))

        np.random.seed(self.seed)
        K.clear_session()
        tf.set_random_seed(self.seed)

        return loss

    def base_test(self, data, sl, hl, lr):
        seq = self.series_to_seq(data, self.max_sl)

        x_pred = seq[:, -(sl + self.num_fc):-self.num_fc].reshape(seq.shape[0], sl, 1)
        y_true = seq[:, -self.num_fc:]
        y_pred = np.zeros([seq.shape[0], self.num_fc])

        for i in range(0, self.bags):
            name = self.rnn_name(sl, hl, lr, i)

            model = self.rnn_load_model(name)
            p = model.predict(x_pred)

            y_pred += p[:, :self.num_fc].reshape(seq.shape[0], self.num_fc)

        y_pred /= float(self.bags)

        return y_pred, y_true

    def base_forecast(self, data, sl, hl, lr):
        fc = self.series_to_fc(data, sl)

        x_fc = fc.reshape(1, sl, 1)
        y_fc = np.zeros(self.num_fc)

        for i in range(0, self.bags):
            name = self.rnn_name(sl, hl, lr, i)

            model = self.rnn_load_model(name)
            p = model.predict(x_fc)

            y_fc += p[:, :self.num_fc].reshape(self.num_fc)

        y_fc /= float(self.bags)

        return y_fc

    def mean_test(self, keys=None):
        if keys is None:
            keys = self.test_pred.keys()

        x_test = []
        for key, val in self.test_pred.items():
            if key in keys:
                x_test.append(val)
        x_test = np.array(x_test)
        x_test = x_test.reshape(x_test.shape[0], (x_test.shape[1] * x_test.shape[2])).transpose()

        y_pred = np.mean(x_test, axis=1)
        y_pred = y_pred.reshape(len(self.test_true), self.num_fc)

        return y_pred

    def mean_forecast(self, name=None, keys=None):
        if keys is None:
            keys = self.fc_pred.keys()

        x_fc = []
        for key, val in self.fc_pred.items():
            if key in keys:
                x_fc.append(val)
        x_fc = np.array(x_fc).transpose()

        y_fc = np.mean(x_fc, axis=1)

        return y_fc

    def meta_train(self, key):
        x_train, y_train = self.get_meta_train()

        for i in range(0, self.bags):
            name = self.meta_name(key, i)
            if self.meta_is_model(name):
                continue

            seed = self.seed + i
            np.random.seed(seed)

            model = self.meta_build_model(key, seed)
            model.fit(x_train, y_train)
            self.meta_save_model(model, name)

        np.random.seed(self.seed)

    def meta_test(self, key):
        x_test = self.get_meta_test()

        n_seq = len(self.test_true)
        y_pred = np.zeros([n_seq, self.num_fc])

        for i in range(0, self.bags):
            name = self.meta_name(key, i)

            model = self.meta_load_model(name)
            p = model.predict(x_test)

            y_pred += p.reshape(n_seq, self.num_fc)

        y_pred /= float(self.bags)

        return y_pred

    def meta_forecast(self, key):
        x_fc = self.get_meta_forecast()
        y_fc = np.zeros(self.num_fc)

        for i in range(0, self.bags):
            name = self.meta_name(key, i)

            model = self.meta_load_model(name)
            p = model.predict(x_fc)

            y_fc += p

        y_fc /= float(self.bags)

        return y_fc

    def plot_train(self, title=None, losses=None, show=True, file=None):
        if title is None:
            title = 'Train'

        if losses is None:
            losses = self.losses

        fig, ax = plt.subplots(figsize=(10, 5))

        colors = plt.get_cmap('tab20')
        epochs = np.arange(1, self.epochs + 1)
        
        for i, name in enumerate(losses):
            ax.plot(epochs, losses[name], label=name, color=colors(i))
        
        ax.set_title(title)
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epoch')
        ax.set_xlim([1, self.epochs])

        if len(losses) > 0:
            ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
        
        if show:
            plt.show()
        if file:
            fig.savefig(self.get_path(file))

    def plot_test(self, title=None, score='rmse', show=True, file=None):
        if title is None:
            title = 'Test'

        colors = plt.get_cmap('tab20')
        labels = []
        values = []
        for key, val in self.scores[score].items():
            labels.append('{} ({:0.2f} | {:0.2f}%)'.format(key, self.scores['rmse'][key], self.scores['mape'][key]))
            values.append(val)
        pos = np.arange(1, len(self.scores[score]) + 1)

        fig, ax = plt.subplots(figsize=(8, 4))

        for i, val in enumerate(values):
            ax.barh(pos[i], val, color=colors(i))

        ax.set_title(title)
        ax.set_yticks(pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()

        if show:
            plt.show()
        if file:
            fig.savefig(self.get_path(file))

    def plot_forecast(self, title=None, keys=None, forecast=False, zoom=1, show=True, file=None):
        if title is None:
            title = 'Forecast'

        data = self.fc_pred if forecast else self.test_pred

        if keys is None:
            keys = data.keys()

        s_data = {}
        for key, val in data.items():
            if key in keys:
                if forecast:
                    sd = self.fc_to_series(val)
                else:
                    sd, _ = self.seq_to_series(val)
                s_data[key] = sd

        colors = plt.get_cmap('tab20')

        fig, ax = plt.subplots(figsize=(15, 5))

        ax.plot(self.series[-(zoom * self.num_fc):], label='true', color='blue', lw=2)
        for i, name in enumerate(s_data):
            sd = s_data[name] if forecast else s_data[name][-1]
            ax.plot(sd, label=name, color=colors(i), lw=1)

        ax.set_title(title)

        if len(s_data) > 0:
            ax.legend(bbox_to_anchor=(1, 1), loc='upper left')

        if show:
            plt.show()
        if file:
            fig.savefig(self.get_path(file))

    def get_path(self, name):
        return '{}/files/ensemble/{}'.format(os.getcwd(), name)

    def rnn_name(self, sl, hl, lr, i=-1):
        name = 'base'

        if len(self.base_params['sl']) > 1:
            name += '_sl_' + str(sl)
        if len(self.base_params['hl']) > 1:
            name += '_hl_' + str(hl)
        if len(self.base_params['lr']) > 1:
            name += '_lr_' + str(lr)
        if i >= 0:
            name += '_i_' + str(i + 1)

        return name

    def rnn_is_model(self, name):
        return os.path.isfile(self.get_path('{}_model.json'.format(name)))

    def rnn_save_model(self, model, name):
        content = model.to_json()
        with open(self.get_path('{}_model.json'.format(name)), 'w') as fh:
            fh.write(content)
        
        model.save_weights(self.get_path('{}_weights.h5'.format(name)))

    def rnn_load_model(self, name):
        with open(self.get_path('{}_model.json'.format(name)), 'r') as fh:
            content = fh.read()
        model = model_from_json(content)
        
        model.load_weights(self.get_path('{}_weights.h5'.format(name)))

        return model

    def rnn_save_history(self, history, name):
        content = json.dumps(history)
        with open(self.get_path('{}_history.json'.format(name)), 'w') as fh:
            fh.write(content)

    def rnn_load_history(self, name):
        with open(self.get_path('{}_history.json'.format(name)), 'r') as fh:
            content = fh.read()
        history = json.loads(content)

        return history

    def meta_name(self, key, i=-1):
        name = 'meta' + '_' + key

        if i >= 0:
            name += '_i_' + str(i + 1)

        return name

    def meta_is_model(self, name):
        return os.path.isfile(self.get_path('{}_model.pml'.format(name)))

    def meta_save_model(self, model, name):
        joblib.dump(model, self.get_path('{}_model.pml'.format(name)))

    def meta_load_model(self, name):
        return joblib.load(self.get_path('{}_model.pml'.format(name)))
