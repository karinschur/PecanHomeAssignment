import os
import pickle
from pathlib import Path
from typing import Dict

from xgboost import XGBRegressor

from utils import utils
from preprocess.data_holder import DataHolder
from preprocess.data_preprocess import Preprocessor


load_from_data = False
preprocessors_names = ['no_lags', 'lags_not_continuous', 'lags_continuous', 'with_nulls']
path = os.path.join(Path(os.getcwd()).parent, 'data')
preprocessors_obj = [Preprocessor(data_path=path, one_hot_encoding=True,
                                  lag_stats=False, remove_outliers=True,
                                  remove_early_data=True),
                     Preprocessor(data_path=path, one_hot_encoding=True,
                                  lag_stats=True, lag_stats_continuous=False, remove_outliers=True,
                                  remove_early_data=True, fill_na=True),
                     Preprocessor(data_path=path, one_hot_encoding=True,
                                  lag_stats=True, lag_stats_continuous=True, remove_outliers=True,
                                  remove_early_data=True, fill_na=True),
                     Preprocessor(data_path=path, one_hot_encoding=True,
                                  lag_stats=True, remove_outliers=True,
                                  remove_early_data=True, fill_na=False)
                     ]

if not load_from_data:
    data_after_preprocessors: Dict[str, DataHolder] = {name: pre() for name, pre in zip(preprocessors_names, preprocessors_obj)}
    for name, data in data_after_preprocessors.items():
        with open(os.path.join(path, f'xgboost_{name}.pkl'), 'wb') as f:
            pickle.dump(data, f)
else:
    preprocessors = []
    for name in preprocessors_names:
        with open(os.path.join(path, f'xgboost_{name}.pkl'), 'rb') as f:
            preprocessors.append(pickle.load(f))
    data_after_preprocessors: Dict[str, DataHolder] = {name: data for name, data in zip(preprocessors_names, preprocessors)}

for (name, data), preprocessor in zip(data_after_preprocessors.items(), preprocessors_obj):
    model = XGBRegressor()
    model.fit(data.train_x, data.train_y,
              eval_metric=['rmse', 'mae'],
              eval_set=[(data.train_x, data.train_y), (data.val_x, data.val_y)],
              verbose=True,
              early_stopping_rounds=10)
    pred_y = model.predict(data.test_x)
    metrics_by_category = utils.results_metrics_by_category(pred_y, data.test_y, preprocessor.data.test)
    utils.print_results_metrics(pred_y, data.test_y)
