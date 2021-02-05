import pandas as pd
from typing import Tuple, List
from pandas import DataFrame as PandasDataFrame


def load_data(path_to_data: str, schema: List[str]
              ) -> Tuple[PandasDataFrame, PandasDataFrame, PandasDataFrame]:
    train_df = read_csv(f'{path_to_data}/train.csv', schema)
    val_df = read_csv(f'{path_to_data}/val.csv', schema)
    test_df = read_csv(f'{path_to_data}/test.csv', schema)

    return train_df, val_df, test_df


def read_csv(path: str, schema: List[str]) -> PandasDataFrame:
    df = pd.read_csv(path)
    if not validate_schema(df, schema):
        raise Exception('Expected different schema')
    return df


def validate_schema(df: PandasDataFrame, schema: List[str]) -> bool:
    return list(df.columns) == schema
