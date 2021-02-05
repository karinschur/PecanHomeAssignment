import pandas as pd
from typing import List, Union
from pandas import DataFrame as PandasDataFrame

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
