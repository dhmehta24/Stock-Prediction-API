import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from nsepy import get_history
from datetime import date
import tensorflow as tf
from django.conf import settings
from tensorflow import keras
import socket
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error,accuracy_score

def predict(name):
    print("Request Recieved for {}".format(name))
    today = date.today()
    try:
        data = get_history(symbol= name, start=date(today.year - 10,today.month, today.day ), end= today)
    except socket.gaierror:
        return "Invalid Stock Ticker or Connection Error"
    #data[['Close']].plot()

    data = data.filter(['Open','Close','High','Low'])

    data = data.filter(['Close'])

    data = pd.DataFrame(data = data)

    print(data.isna().any())

    pivot = int(len(data) * 0.8)

    training_data = data.iloc[:pivot]
    testing_data = data.iloc[pivot:]

    print("testing data",testing_data)

    past_60 = training_data.tail(60)
    test_data = past_60.append(testing_data)

    scaler = MinMaxScaler(feature_range=(0, 1))

    scaled_training_data = scaler.fit_transform(training_data.values.reshape(-1,1))
    scaled_testing_data = scaler.transform(test_data.values.reshape(-1,1))
    scaled_final_data = scaler.transform(testing_data)

    data = scaler.transform(data)

    x_test, y_test = [], []

    #print("scaled shape",scaled_testing_data.shape[0])

    for i in range(60,scaled_testing_data.shape[0]):
        x_test.append(scaled_testing_data[i - 60:i])
        y_test.append(scaled_testing_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)

    y_test = y_test.reshape(-1,1)

    print("y_test value")

    print(x_test.shape, y_test.shape)

    model = keras.models.load_model('stock_prediction_model')


    predictions = model.predict(x_test)

    n_data = [data[len(data)+1 - 61:len(data+1),0]]

    n_data = np.array(n_data)

    print(n_data.shape)

    n_data = np.reshape(n_data,(n_data.shape[0], n_data.shape[1], 1))

    predi = model.predict(n_data)

    predi_result = scaler.inverse_transform(predi)

    #result = predi_result.astype(np.int)

    return predi_result[0][0]



