#!/usr/bin/env python

import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
from sklearn.externals import joblib

from TSEnsemble import TSEnsemble

SEED = 42
np.random.seed(SEED)

# Set currency
currency = 'cop'

# Set params
base_params = {
    'sl': [60, 70, 80],
    'hl': [2, 3],
    'lr': [.01]
}
meta_params = {
    'rf': {
        'n_estimators': 100
    },
    'ridge': {
        'alpha': 1.
    },
    'xgb': {
        'n_estimators': 1000,
        'max_depth': 8,
        'learning_rate': .01
    }
}
diff = 1
bags = 3
epochs = 30
hold_split = .3
test_split = .3
verbose = 2

# Set series range
dt_from = '2012-01-01'
dt_to = '2018-11-27'

print('START')

# Read data
df = pd.read_csv(f'{os.getcwd()}/docs/sample-fx.csv', sep=',')

# Parse currency and set time series
y = df[df['currency'] == currency.upper()].drop(['currency'], axis=1).fillna(0).set_index('rated_at').rename_axis(None)
y.index = pd.to_datetime(y.index)
y = y['value'].sort_index()[dt_from:dt_to]

print('Total: {:d}'.format(len(y)))

# Initialize model
model = TSEnsemble(bags=bags, diff=diff, epochs=epochs, seed=SEED, verbose=verbose)

networks, num_fc, max_sl = model.set_params(base_params, meta_params)
dtrain, dmeta, dtest = model.init_series(y, hold_split=hold_split, test_split=test_split)

print('Networks: {:d} | Forecast: {:d} | Maxlen: {:d}'.format(networks, num_fc, max_sl))
print('Train: {:d} | Meta: {:d} | Test: {:d}'.format(len(dtrain), len(dmeta), len(dtest)))

# Train model
model.model_train()

model.plot_train(file='losses.png', show=False)

# Test model
model.model_test()

best, scores = model.best_model()
models = model.get_models()
corr = model.get_mean_corr()

print('Mean Pearson Corr: {:0.2f}%'.format(corr * 100))
print('Best Model: {} ({:0.2f} | {:0.2f}%)'.format(best, scores['rmse'], scores['mape']))
model.plot_forecast(title='Base', keys=models[:networks], file='base_test.png', show=False)
model.plot_forecast(title='Meta', keys=models[networks:], file='meta_test.png', show=False)
model.plot_test(score='rmse', file='scores.png', show=False)

# Forecast model
model.model_forecast()

model.plot_forecast(title='Base', keys=models[:networks], forecast=True, file='base_forecast.png', show=False)
model.plot_forecast(title='Meta', keys=models[networks:], forecast=True, file='meta_forecast.png', show=False)

# Save model
joblib.dump(model, '{}/files/{}'.format(os.getcwd(), 'ensemble.pml'))

print('END')
