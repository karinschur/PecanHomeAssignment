from abc import ABC, abstractmethod
from pandas import DataFrame as PandasDataFrame

from preprocess.data_holder import DataHolder


class BaseModel(ABC):

    def __init__(self, *, data: DataHolder):
        self.data = data

    @abstractmethod
    def fit(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self, pred_df: PandasDataFrame):
        raise NotImplementedError
