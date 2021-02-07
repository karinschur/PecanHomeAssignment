from typing import Dict

import lightgbm as lgb
import os
import pickle
from pathlib import Path
import pickle
import numpy as np

from preprocess.data_holder import DataHolder
from utils import constants,utils
from preprocess.data_preprocess import Preprocessor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,  mean_absolute_percentage_error


load_from_data = False
preprocessors_names = ['no_lags', 'lags_not_continuous', 'lags_continuous', 'with_nulls']
path = os.path.join(Path(os.getcwd()).parent, 'data')
preprocessors_obj = [Preprocessor(data_path=path, one_hot_encoding=False,
                                  lag_stats=False, remove_outliers=True,
                                  remove_early_data=True),
                     Preprocessor(data_path=path, one_hot_encoding=False,
                                  lag_stats=True, lag_stats_continuous=False, remove_outliers=True,
                                  remove_early_data=True, fill_na=True),
                     Preprocessor(data_path=path, one_hot_encoding=False,
                                  lag_stats=True, lag_stats_continuous=True, remove_outliers=True,
                                  remove_early_data=True, fill_na=True),
                     Preprocessor(data_path=path, one_hot_encoding=False,
                                  lag_stats=True, remove_outliers=True,
                                  remove_early_data=True, fill_na=False)
                     ]

if not load_from_data:
    data_after_preprocessors: Dict[str, DataHolder] = {name: pre() for name, pre in zip(preprocessors_names, preprocessors_obj)}
    for name, data in data_after_preprocessors.items():
        with open(os.path.join(path, f'light_gbm_{name}.pkl'), 'wb') as f:
            pickle.dump(data, f)
else:
    preprocessors = []
    for name in preprocessors_names:
        with open(os.path.join(path, f'light_gbm_{name}.pkl'), 'rb') as f:
            preprocessors.append(pickle.load(f))
    data_after_preprocessors: Dict[str, DataHolder] = {name: data for name, data in zip(preprocessors_names, preprocessors)}


for (name, data), preprocessor in zip(data_after_preprocessors.items(), preprocessors_obj):

    categorical_features = [constants.category_header]

    train_ds = lgb.Dataset(data.train_x, label=data.train_y, categorical_feature=categorical_features, free_raw_data=False)
    val_ds = lgb.Dataset(data.val_x, label=data.val_y, categorical_feature=categorical_features, free_raw_data=False)

    watchlist = [train_ds, val_ds]
    evals_result = {}

    params = {
        "objective": "regression",
        "boosting": "gbdt",
        "num_leaves": 100,
        "learning_rate": 0.05,
        "feature_fraction": 0.85,
        "reg_lambda": 2,
        "metric": ['rmse', 'mae']
    }

    model = lgb.train(params, train_set=train_ds, num_boost_round=1000, valid_sets=watchlist, evals_result=evals_result,
                      verbose_eval=25, early_stopping_rounds=200)
    pred_y = model.predict(data.test_x, num_iteration=model.best_iteration)

    _ = lgb.plot_metric(evals_result, metric='rmse')
    plt.show()

    metrics_by_category = utils.results_metrics_by_category(pred_y, data.test_y, preprocessor.data.test)
    utils.print_results_metrics(pred_y, data.test_y)
