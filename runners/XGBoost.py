import xgboost as xgb
from xgboost import plot_importance
import os
from pathlib import Path
import numpy as np
from pandas import read_csv
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,  mean_absolute_percentage_error
from preprocess.data_preprocess import Preprocessor
import time
import plotly.express as px
import matplotlib as plt



def plot_features(booster, figsize):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    return plot_importance(booster=booster, ax=ax)


path = os.path.join(Path(os.getcwd()).parent, 'data')
preprocessor_no_lags = Preprocessor(data_path=path, one_hot_encoding=True,
                                    lag_stats=False, remove_outliers=True,
                                    remove_early_data=True)
preprocessor_lags_not_continuous = Preprocessor(data_path=path, one_hot_encoding=True,
                                                lag_stats=True, lag_stats_continuous=False, remove_outliers=True,
                                                remove_early_data=True, fill_na=True)
preprocessor_lags_continuous = Preprocessor(data_path=path, one_hot_encoding=True,
                                            lag_stats=True, lag_stats_continuous=True, remove_outliers=True,
                                            remove_early_data=True, fill_na=True)
preprocessor_with_nulls = Preprocessor(data_path=path, one_hot_encoding=True,
                                       lag_stats=True, remove_outliers=True,
                                       remove_early_data=True, fill_na=False)
preprocessors = {'no_lags': preprocessor_no_lags, 'lags_not_continuous': preprocessor_lags_not_continuous,
                 'lags_continuous': preprocessor_lags_continuous, 'with_nulls': preprocessor_with_nulls}

for name, preprocessor in preprocessors.items():
    # Preprocess: feature extraction + feature engineering
    data_after_preprocess = preprocessor()

    X_train = data_after_preprocess.train.drop('Label', axis=1)
    X_val = data_after_preprocess.val.drop('Label', axis=1)
    y_train = data_after_preprocess.train['Label']
    y_val = data_after_preprocess.val['Label']

    xg_train = xgb.DMatrix(data=X_train, label=y_train)

    xgb_params = {'objective':'reg:squarederror',
        'colsample_bytree':0.3,
        'learning_rate':0.1,
        'min_child_weight': 3,
        'max_depth':7,
        'alpha':10,
        'subsample':0.8,
        'gamma':0.005,
        'eta':0.1,
        'seed':42}

    xg_reg = xgb.XGBRegressor(xgb_params)
    # do cross validation
    print('Start cross validation')
    cv_result = xgb.cv(xgb_params, xg_train, num_boost_round=50, nfold=5, metrics=['rmse', 'mae'],
                       early_stopping_rounds=50, stratified=True, seed=1101)
    print('Best number of trees = {}'.format(cv_result.shape[0]))
    xg_reg.set_params(n_estimators=cv_result.shape[0])
    print('Fit on the training data')
    xg_reg.fit(X_train, y_train, eval_metric=['rmse', 'mae'])

    print('Predict the probabilities based on features in the test set')
    preds_train = xg_reg.predict(X_train)
    preds_val = xg_reg.predict(X_val)
    # preds_val = xg_reg.predict_proba(X_val, ntree_limit=cv_result.shape[0])



    # res = xg_reg.fit(
    #     X_train,
    #     y_train,
    #     eval_metric="rmse",
    #     eval_set=[(X_train, y_train), (X_val, y_val)],
    #     verbose=True,
    #     early_stopping_rounds=40,
    #     )



    rmse = np.sqrt(mean_squared_error(y_val, preds_val))
    print("RMSE: %f" % (rmse))
    rmse = mean_absolute_error(y_val, preds_val)
    print("MAE: %f" % (rmse))
    mape = mean_absolute_percentage_error(y_val, preds_val)
    print("MAPE: %f" % (mape))
    r2 = r2_score(y_val, preds_val)
    print("R2: %f" % (r2))





    print((cv_results["test-rmse-mean"]).tail(1))
    fig = px.scatter(x=preds_val, y=y_val, title=f'Validation performance VS true label using preprocessor {name}')
    fig.show()

    plot_features(xg_reg, (10, 14))

