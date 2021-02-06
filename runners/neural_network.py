import keras
from keras.models import Sequential
from keras.layers import Dense
import os
from pathlib import Path
import plotly.express as px
import pickle

from utils import constants
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

train_x = data_after_preprocess.train.drop('Label', axis=1)
train_y = data_after_preprocess.train['Label']

val_x = data_after_preprocess.val.drop('Label', axis=1)
val_y = data_after_preprocess.val['Label']

test_x = data_after_preprocess.test.drop('Label', axis=1)
test_y = data_after_preprocess.test['Label']

model = Sequential()
model.add(Dense(400, input_dim=len(train_x.columns), activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss=keras.losses.mean_absolute_error, optimizer='adam', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=10, batch_size=2000)

_, accuracy = model.evaluate(val_x, val_y)
predictions = model.predict(test_x)
fig = px.scatter(x=predictions, y=test_y)
fig.show()
print()
