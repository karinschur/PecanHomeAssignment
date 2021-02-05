from typing import Union
from pandas import DataFrame as PandasDataFrame

from utils import utils


class DataHolder:

    def __init__(self, *, train: Union[str, PandasDataFrame],
                 val: Union[str, PandasDataFrame],
                 test: Union[str, PandasDataFrame]):
        self.train = utils.load_df(train)
        self.val = utils.load_df(val)
        self.test = utils.load_df(test)
