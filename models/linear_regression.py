from pandas import DataFrame as PandasDataFrame
from sklearn.linear_model import LinearRegression

from utils import constants
from models.base_model import BaseModel
from preprocess.data_holder import DataHolder


class LinearRegressionModel(BaseModel):

    def __init__(self, *, data: DataHolder):
        super().__init__(data=data)
        self.linear_regression = LinearRegression()

    def fit(self):
        train_x = self.data.train.drop('Label', axis=1)  # values converts it into a numpy array
        train_y = self.data.train['Label']  # -1 means that calculate the dimension of rows, but have 1 column
        self.linear_regression.fit(train_x, train_y)  # perform linear regression

    def predict(self, pred_df: PandasDataFrame):
        if constants.label_header in pred_df.columns:
            pred_df = pred_df.drop(constants.label_header, axis=1)

        return self.linear_regression.predict(pred_df)
