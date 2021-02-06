import pandas as pd
import numpy as np
import os
import pickle
from pathlib import Path
from typing import List, Union
from pandas import DataFrame as PandasDataFrame, Series as PandasSeries
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,  mean_absolute_percentage_error
from utils import constants


def load_df(df: Union[str, PandasDataFrame]) -> PandasDataFrame:
    return read_csv(path=df, schema=constants.schema) \
        if isinstance(df, str) else df


def read_csv(path: str, schema: List[str]) -> PandasDataFrame:
    df = pd.read_csv(path)
    if not validate_schema(df, schema):
        raise Exception('Expected different schema')
    return df


def validate_schema(df: PandasDataFrame, schema: List[str]) -> bool:
    return list(df.columns) == schema


def results_metrics_by_category(pred_y: np.array, actual_y: PandasSeries,
                                original_df: PandasDataFrame) -> PandasDataFrame:
    result_df = pd.DataFrame(actual_y)
    result_df['pred'] = pred_y
    initial_data = original_df.set_index(constants.pecan_id_header)
    result_df = result_df.merge(initial_data[constants.category_header], left_index=True, right_index=True)

    def metric_by_category(df):
        r2 = r2_score(df[constants.label_header], df['pred'])
        mape = mean_absolute_percentage_error(df[constants.label_header], df['pred'])
        mae = mean_absolute_error(df[constants.label_header], df['pred'])
        rmse = np.sqrt(mean_squared_error(df[constants.label_header], df['pred']))
        return pd.DataFrame(dict(r2=r2, mape=mape, mae=mae, rmse=rmse))

    return result_df.groupby(constants.category_header).apply(metric_by_category).reset_index()


def print_results_metrics(pred_y: np.array, actual_y: PandasSeries):
    print(f'RMSE: {np.sqrt(mean_squared_error(actual_y, pred_y))}')
    print(f'MAE: {mean_absolute_error(actual_y, pred_y)}')
    print(f'MAPE: {mean_absolute_percentage_error(actual_y, pred_y)}')
    print(f'R2: {r2_score(actual_y, pred_y)}')
