import os
from pathlib import Path
import plotly.express as px

from utils import constants
from preprocess.data_preprocess import Preprocessor
from models.linear_regression import LinearRegressionModel

# Preprocess: feature extraction + feature engineering
path = os.path.join(Path(os.getcwd()).parent, 'data')
preprocessor = Preprocessor(data_path=path, one_hot_encoding=True,
                            lag_stats=True, remove_outliers=True,
                            remove_early_data=True)
data_after_preprocess = preprocessor()

# TODO: Data visualizations

# Model training
linear_regression_model = LinearRegressionModel(data=data_after_preprocess)
linear_regression_model.fit()

# Prediction
pred_y_val = linear_regression_model.predict(data_after_preprocess.val)
pred_y_test = linear_regression_model.predict(data_after_preprocess.test)

# Plots
fig = px.scatter(x=pred_y_val, y=data_after_preprocess.val[constants.label_header])
fig.show()
fig = px.scatter(x=pred_y_test, y=data_after_preprocess.test[constants.label_header])
fig.show()
