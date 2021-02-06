from typing import Union
from pandas import DataFrame as PandasDataFrame

from utils import utils, constants


class DataHolder:

    def __init__(self, *, train: Union[str, PandasDataFrame],
                 val: Union[str, PandasDataFrame],
                 test: Union[str, PandasDataFrame]):
        self.train = utils.load_df(train)
        self.val = utils.load_df(val)
        self.test = utils.load_df(test)

    @property
    def train_x(self):
        return self.train.drop(constants.label_header, axis=1)

    @property
    def train_y(self):
        return self.train[constants.label_header]

    @property
    def val_x(self):
        return self.val.drop(constants.label_header, axis=1)

    @property
    def val_y(self):
        return self.val[constants.label_header]

    @property
    def test_x(self):
        return self.test.drop(constants.label_header, axis=1)

    @property
    def test_y(self):
        return self.test[constants.label_header]
