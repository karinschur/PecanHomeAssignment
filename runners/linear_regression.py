import os
from pathlib import Path
import plotly.express as px

from utils import constants
from preprocess.data_preprocess import Preprocessor
from models.linear_regression import LinearRegressionModel


path = os.path.join(Path(os.getcwd()).parent, 'data')
preprocessor = Preprocessor(data_path=path, one_hot_encoding=True,
                            lag_stats=True, remove_outliers=True,
                            remove_early_data=True)
data_after_preprocess = preprocessor()

linear_regression_model = LinearRegressionModel(data=data_after_preprocess)
linear_regression_model.fit()
pred_y = linear_regression_model.predict(data_after_preprocess.val)

fig = px.scatter(x=pred_y, y=data_after_preprocess.val[constants.label_header])
fig.show()

print()
