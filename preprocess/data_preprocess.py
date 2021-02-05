import pandas as pd
from typing import List, Union
from pandas import DataFrame as PandasDataFrame

from utils import constants
from preprocess.data_holder import DataHolder


class Preprocessor:

    def __init__(self, *, data_path: str, one_hot_encoding: bool = False,
                 remove_outliers: bool = False, remove_early_data: bool = False,
                 lag_stats: bool = False, window_sizes: List[str] = None,
                 one_hot_cols: Union[str, List[str]] = None):
        self.data = DataHolder(path_to_data=data_path)
        self.remove_outliers = remove_outliers
        self.one_hot_encoding = one_hot_encoding
        self.remove_early_data = remove_early_data
        self.lag_stats = lag_stats
        self.window_sizes = window_sizes or constants.window_sizes
        self.one_hot_cols = one_hot_cols or constants.one_hot_cols

    def __call__(self, *args, **kwargs) -> DataHolder:
        self.data.train = self.basic_feature_extraction(self.data.train)
        self.data.val = self.basic_feature_extraction(self.data.val)
        self.data.test = self.basic_feature_extraction(self.data.test)

        if self.remove_outliers:
            outlier_ids = [17179869190, 51539608572, 60129542923]
            self.data.train = self.data.train[(~self.data.train[constants.pecan_id_header].isin((outlier_ids)))]
            self.data.train = self.data.train[(self.data.train[constants.label_header] >= 0)]

        if self.lag_stats:
            self.data.train = self.create_lag_features(self.data.train)
            self.data.val = self.create_lag_features(self.data.val)
            self.data.val = self.create_lag_features(self.data.val)

        if self.one_hot_encoding:
            self.one_hot_encoded()

        if self.remove_early_data:
            self.data.train = self.data.train[self.data.train['Year'] >= 2016]

        self.data.train = self.data.train.drop([constants.marker_header, 'Year'], axis=1)
        self.data.train.set_index(constants.pecan_id_header)
        self.data.val.set_index(constants.pecan_id_header)
        self.data.test.set_index(constants.pecan_id_header)

        return self.data

    @staticmethod
    def basic_feature_extraction(df: PandasDataFrame) -> PandasDataFrame:
        unique_dates = sorted(df[constants.marker_header].unique())
        date_num_mapping = {date: i for i, date in enumerate(unique_dates)}
        df[constants.date_num_header] = df[constants.marker_header].apply(lambda date: date_num_mapping[date])

        # I've seen there is a consiquencial appearance of category since it first appears.
        #  Category start appearing at different points. A good feature would be num of appearances.
        count_category_appearances = df.groupby(constants.category_header)[constants.marker_header].count()
        count_category_appearances.name = constants.category_appearance_header
        df = df.merge(count_category_appearances, left_on=constants.category_header, right_index=True)

        # Separate Datetime values
        df[constants.marker_header] = pd.to_datetime(pd.Series(df[constants.marker_header]),
                                                     format=constants.marker_format)
        df['Month'] = df[constants.marker_header].apply(lambda dt: dt.month)
        df['Year'] = df[constants.marker_header].apply(lambda dt: dt.year)
        return df

    def one_hot_encoded(self):
        self.data.train = self.apply_one_hot_encoding(self.data.train,
                                                      categorical_cols=self.one_hot_cols)
        self.data.val = self.apply_one_hot_encoding(self.data.val,
                                                    categorical_cols=self.one_hot_cols)
        self.data.test = self.apply_one_hot_encoding(self.data.test,
                                                     categorical_cols=self.one_hot_cols)
        train_cols = set(self.data.train.columns)
        val_missing_cols = list(train_cols - set(self.data.val.columns))
        test_missing_cols = list(train_cols - set(self.data.test.columns))

        self.data.val[val_missing_cols] = 0
        self.data.test[test_missing_cols] = 0

    @staticmethod
    def apply_one_hot_encoding(df: PandasDataFrame, *,
                               categorical_cols: Union[str, List[str]]
                               ) -> PandasDataFrame:
        one_hot_cat = pd.get_dummies(df.set_index(constants.pecan_id_header)[categorical_cols],
                                     prefix=constants.pecan_id_header)
        df = df.drop(categorical_cols, axis=1)
        return df.merge(one_hot_cat, left_on=constants.pecan_id_header, right_index=True)

    def create_lag_features(self, df: PandasDataFrame):
        window_sizes = self.window_sizes

        def create_lag_features(applied_df: PandasDataFrame):
            d = {constants.date_num_header: applied_df[constants.date_num_header]}
            label_after_sort = applied_df.sort_values(constants.date_num_header).Label
            for lag in window_sizes:
                temp_sum = label_after_sort.rolling(lag).sum().shift()
                d[f'label_sum_lag_{lag}_months'] = temp_sum
                temp_avg = label_after_sort.rolling(lag).mean().shift()
                d[f'label_avg_lag_{lag}_months'] = temp_avg
                temp_std = label_after_sort.rolling(lag).std().shift()
                d[f'label_std_lag_{lag}_months'] = temp_std
                if lag == 2:
                    temp_diff = applied_df.sort_values(constants.date_num_header).Label.diff()
                    d[f'label_diff_lag_{lag}_months'] = temp_diff

            return pd.DataFrame(d)

        df_with_lags = df.groupby(constants.category_header).apply(create_lag_features)
        df_with_lags = df_with_lags.reset_index(constants.category_header)
        return df.merge(df_with_lags, on=[constants.category_header, constants.date_num_header])


if __name__ == '__main__':
    import os
    from pathlib import Path
    path = os.path.join(Path(os.getcwd()).parent, 'data')
    x = Preprocessor(data_path=path, one_hot_encoding=True,
                     lag_stats=True, remove_outliers=True,
                     remove_early_data=True)
    dh = x()
    print()
