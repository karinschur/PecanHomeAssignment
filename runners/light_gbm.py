import os
import pickle
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt

from utils import constants
from preprocess.data_preprocess import Preprocessor


load_data_from_file = False
path = os.path.join(Path(os.getcwd()).parent, 'data')
if load_data_from_file:
    with open(os.path.join(path, 'data_after_preprocess_no_one_hot.pkl'), 'rb') as f:
        data_after_preprocess = pickle.load(f)
else:
    preprocessor = Preprocessor(data_path=path, one_hot_encoding=False,
                                lag_stats=True, lag_stats_continuous=True,
                                remove_outliers=True, remove_early_data=True,
                                fill_na=True)
    data_after_preprocess = preprocessor()

    with open(os.path.join(path, 'data_after_preprocess_no_one_hot.pkl'), 'wb') as f:
        pickle.dump(data_after_preprocess, f)

train_x = data_after_preprocess.train.drop('Label', axis=1)
train_y = data_after_preprocess.train['Label']

val_x = data_after_preprocess.val.drop('Label', axis=1)
val_y = data_after_preprocess.val['Label']

test_x = data_after_preprocess.test.drop('Label', axis=1)
test_y = data_after_preprocess.test['Label']

categorical_features = [constants.category_header]

train_ds = lgb.Dataset(train_x, label=train_y, categorical_feature=categorical_features, free_raw_data=False)
val_ds = lgb.Dataset(train_x, label=train_y, categorical_feature=categorical_features, free_raw_data=False)

watchlist = [train_ds, val_ds]
evals_result = {}

params = {
    "objective": "regression",
    "boosting": "gbdt",
    "num_leaves": 100,
    "learning_rate": 0.05,
    "feature_fraction": 0.85,
    "reg_lambda": 2,
    "metric": "rmse"
}

print("Building model with first half and validating on second half:")
model_half_1 = lgb.train(params, train_set=train_ds, num_boost_round=1000, valid_sets=watchlist, evals_result=evals_result, verbose_eval=25, early_stopping_rounds=200)

_ = lgb.plot_metric(evals_result, metric='rmse')
plt.show()
