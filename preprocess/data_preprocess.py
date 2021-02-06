import pandas as pd
from typing import List, Union, Tuple
from pandas import DataFrame as PandasDataFrame

from utils import constants
from preprocess.data_holder import DataHolder


class Preprocessor:

    def __init__(self, *, data_path: str, one_hot_encoding: bool = False,
                 remove_outliers: bool = False, remove_early_data: bool = False,
                 lag_stats: bool = False, window_sizes: List[str] = None,
                 one_hot_cols: Union[str, List[str]] = None):
        self.data = DataHolder(train=f'{data_path}/train.csv',
                               val=f'{data_path}/val.csv',
                               test=f'{data_path}/test.csv')
        self.remove_outliers = remove_outliers
        self.one_hot_encoding = one_hot_encoding
        self.remove_early_data = remove_early_data
        self.lag_stats = lag_stats
        self.window_sizes = window_sizes or constants.window_sizes
        self.one_hot_cols = one_hot_cols or constants.one_hot_cols

    def __call__(self, *args, **kwargs) -> DataHolder:
        train, last_index = self.create_date_num_col(self.data.train)
        val, last_index = self.create_date_num_col(self.data.val, last_index)
        test, _ = self.create_date_num_col(self.data.test, last_index)
        train, val, test = [self.basic_feature_extraction(df) for df in [train, val, test]]

        if self.remove_outliers:
            train = train[(~train[constants.pecan_id_header].isin(constants.outlier_ids))]
            train, val, test = [df[df[constants.label_header] >= 0] for df in [train, val, test]]

        if self.lag_stats:
            # train, val, test = [self.create_lag_features(df) for df in [train, val, test]]
            train, val, test = self.correct_lags(train=train, val=val, test=test)

        if self.one_hot_encoding:
            train, val, test = self.one_hot_encoded(train=train, val=val, test=test)

        if self.remove_early_data:
            train = train[train['Year'] >= 2016]

        train, val, test = [df.set_index(constants.pecan_id_header) for df in [train, val, test]]
        train, val, test = [df.drop([constants.marker_header, 'Year'], axis=1) for df in [train, val, test]]

        test = test[train.columns]
        val = val[train.columns]

        return DataHolder(train=train, val=val, test=test)

    @staticmethod
    def basic_feature_extraction(df: PandasDataFrame) -> PandasDataFrame:
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

    @staticmethod
    def create_date_num_col(df: PandasDataFrame, last_index: int = 0
                            ) -> Tuple[PandasDataFrame, int]:
        unique_dates = sorted(df[constants.marker_header].unique())
        date_num_mapping = {date: i + last_index for i, date in enumerate(unique_dates)}
        df[constants.date_num_header] = df[constants.marker_header].apply(lambda date: date_num_mapping[date])
        return df, len(unique_dates) + last_index

    def one_hot_encoded(self, *, train: PandasDataFrame,
                        val: PandasDataFrame,
                        test: PandasDataFrame
                        ) -> Tuple[PandasDataFrame, PandasDataFrame, PandasDataFrame]:
        train, val, test = [self.apply_one_hot_encoding(df, categorical_cols=self.one_hot_cols)
                            for df in [train, val, test]]
        train_cols = set(train.columns)
        val_missing_cols = list(train_cols - set(val.columns))
        test_missing_cols = list(train_cols - set(test.columns))

        val[val_missing_cols] = 0
        test[test_missing_cols] = 0

        return train, val, test

    def correct_lags(self, *, train: PandasDataFrame,
                     val: PandasDataFrame,
                     test: PandasDataFrame
                     ) -> Tuple[PandasDataFrame, PandasDataFrame, PandasDataFrame]:
        train_return_df = self.create_lag_features(train)
        val_return_df = self.align_following_dfs(first_df=train, second_df=val)
        test_return_df = self.align_following_dfs(first_df=val, second_df=test)
        return train_return_df, val_return_df, test_return_df

    def align_following_dfs(self, first_df: PandasDataFrame,
                            second_df: PandasDataFrame) -> PandasDataFrame:
        max_window_size = max(self.window_sizes)
        first_max_date_num = max(first_df[constants.date_num_header])
        second_df_categories = second_df[constants.category_header].unique()
        first_df = first_df[first_df[constants.category_header].isin(second_df_categories)]
        max_date_by_category = first_df.groupby(constants.category_header).agg({constants.date_num_header: 'max'})
        max_date_by_category = max_date_by_category[
            max_date_by_category[constants.date_num_header] == first_max_date_num]
        valid_categories = list(max_date_by_category.index)
        first_df = first_df[first_df[constants.category_header].isin(valid_categories)]
        date_num_count = first_df.groupby(constants.category_header).agg({constants.date_num_header: 'count'})
        date_num_count = date_num_count[date_num_count[constants.date_num_header] >= max_window_size]
        valid_categories = list(date_num_count.index)
        first_df = first_df[first_df[constants.category_header].isin(valid_categories)]
        first_df = first_df[first_df[constants.date_num_header].isin(range(first_max_date_num + 1)[-max_window_size:])]
        first_df = first_df.append(second_df[second_df[constants.date_num_header] == first_max_date_num + 1])
        first_df = self.create_lag_features(first_df)
        second_first_date = first_df[first_df[constants.date_num_header] == first_max_date_num + 1]

        return_df = self.create_lag_features(second_df)
        return_df = return_df[return_df[constants.date_num_header] > first_max_date_num + 1]
        return_df = return_df.append(second_first_date)

        return return_df

    @staticmethod
    def apply_one_hot_encoding(df: PandasDataFrame, *,
                               categorical_cols: Union[str, List[str]]
                               ) -> PandasDataFrame:
        one_hot_cat = pd.get_dummies(df.set_index(constants.pecan_id_header)[categorical_cols],
                                     prefix=constants.category_header)
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
        df = df.merge(df_with_lags, on=[constants.category_header, constants.date_num_header])

        for lag in self.window_sizes:
            df[f'label_sum_lag_{lag}_months'] = df[f'label_sum_lag_{lag}_months'].fillna(df['Label'])
            df[f'label_avg_lag_{lag}_months'] = df[f'label_avg_lag_{lag}_months'].fillna(df['Label'])
            df[f'label_std_lag_{lag}_months'] = df[f'label_std_lag_{lag}_months'].fillna(df['Label'])
            if lag == 2:
                df[f'label_diff_lag_{lag}_months'] = df[f'label_diff_lag_{lag}_months'].fillna(df['Label'])

        return df


if __name__ == '__main__':
    import os
    from pathlib import Path
    path = os.path.join(Path(os.getcwd()).parent, 'data')
    x = Preprocessor(data_path=path, one_hot_encoding=True,
                     lag_stats=True, remove_outliers=True,
                     remove_early_data=True)
    dh = x()
    print()
