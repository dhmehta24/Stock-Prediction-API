import numpy as np
import pandas as pd
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from nsepy import get_history
from datetime import date
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error

def build_model():

    today = date.today()
    data = get_history(symbol= 'MRF', start=date(today.year - 10 ,today.month, today.day ), end= today )
    # data[['Close']].plot()

    data = data.filter(['Close'])

    pivot = date(today.year - 2 ,today.month, today.day)

    training_data = data[data.index < pivot]
    testing_data = data[data.index >= pivot]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_training_data = scaler.fit_transform(training_data.values.reshape(-1,1))

    X_train = []
    y_train = []

    for i in range(60, scaled_training_data.shape[0]):
        X_train.append(scaled_training_data[i-60:i])
        y_train.append(scaled_training_data[i,0])

    X_train, y_train = np.array(X_train), np.array(y_train)

    X_train = np.reshape(X_train,(X_train.shape[0], X_train.shape[1], 1))

    print(X_train.shape[1])

    model = Sequential()

    model.add(LSTM(units = 64, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    model.add(Dropout(0.2))

    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))

    model.add(Dense(units = 1))

    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

    model.evaluate(X_train,y_train)

    model.save('stock_prediction_model')


