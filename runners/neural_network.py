import os
import pickle
from pathlib import Path

import keras
import plotly.express as px
from keras.layers import Dense
from keras.models import Sequential

from preprocess.data_preprocess import Preprocessor


load_data_from_file = True
path = os.path.join(Path(os.getcwd()).parent, 'data')
if load_data_from_file:
    with open(os.path.join(path, 'data_after_preprocess.pkl'), 'rb') as f:
        data_after_preprocess = pickle.load(f)
else:
    preprocessor = Preprocessor(data_path=path, one_hot_encoding=True,
                                lag_stats=True, lag_stats_continuous=True,
                                remove_outliers=True, remove_early_data=True,
                                fill_na=True)
    data_after_preprocess = preprocessor()

    with open(os.path.join(path, 'data_after_preprocess.pkl'), 'wb') as f:
        pickle.dump(data_after_preprocess, f)

model = Sequential()
model.add(Dense(400, input_dim=len(data_after_preprocess.train_x.columns), activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss=keras.losses.mean_absolute_error, optimizer='adam', metrics=['accuracy'])
model.fit(data_after_preprocess.train_x, data_after_preprocess.train_y, epochs=10, batch_size=2000)

_, accuracy = model.evaluate(data_after_preprocess.val_x, data_after_preprocess.val_y)
predictions = model.predict(data_after_preprocess.test_x)
fig = px.scatter(x=predictions, y=data_after_preprocess.test_y)
fig.show()
