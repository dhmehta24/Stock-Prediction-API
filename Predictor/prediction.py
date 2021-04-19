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
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error,accuracy_score

def import_data(name):
    today = date.today()
    data = get_history(symbol= name, start=date(today.year - 10,today.month, today.day ), end= today )
    #data[['Close']].plot()

    data = data.filter(['Open','Close','High','Low'])

    data = data.filter(['Close'])

    #df = pd.DataFrame(df)

    #df.plot()

    pivot = date(today.year - 5,today.month, today.day)

    training_data = data[data.index < pivot]
    testing_data = data[data.index >= pivot]

    print("testing data",testing_data)

    past_60 = training_data.tail(60)
    test_data = past_60.append(testing_data)

    #print(test_data)

    scaler = MinMaxScaler(feature_range=(0, 1))

    scaled_training_data = scaler.fit_transform(training_data)
    scaled_testing_data = scaler.transform(test_data.values.reshape(-1,1))
    scaled_final_data = scaler.transform(testing_data)

    data = scaler.transform(data)

    #print(scaled_testing_data)

    x_test, y_test = [], []

    print("scaled shape",scaled_testing_data.shape[0])

    for i in range(60,scaled_testing_data.shape[0]):
        x_test.append(scaled_testing_data[i - 60:i])
        y_test.append(scaled_testing_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)

    #x_test = x_test.reshape(-1,1)

    y_test = y_test.reshape(-1,1)

    print("y_test value")

    print(x_test.shape, y_test.shape)

    model = keras.models.load_model(settings.MODEL)

    #model = tf.lite.TFLiteConverter.from_saved_model(settings.MODEL,'saved_model.pb')

    predictions = model.predict(x_test)

    #predictions.reshape(1238,60,4)

    #predictions = scaler.inverse_transform(predictions)

    #print(predictions)

    #score = accuracy_score(testing_data['Close'],predictions)

    #print("Accuracy",score * 100)

    dividers = scaler.scale_

    open_div = 1/dividers[0]
    #close_div = 1/dividers[3]

    pred_open = predictions[:,0] * open_div
    #pred_close = predictions[:,1] * open_div

    print(pred_open)
    #print(pred_close)

    actual_open = y_test[:,0] * open_div
    #actual_close = scaler.inverse_transform(y_test[:,1])

    new_open_df = pd.DataFrame({"Actual Open": actual_open,"Predicted Open":pred_open})
    #new_close_df = pd.DataFrame({"Actual Close": actual_close, "Predicted Close": pred_close})
    print(new_open_df)
    #print(new_close_df)

    """sns.set(style='darkgrid')
    plt.figure(figsize=(15, 6))

    plt.title("{} Stock Market Close Price Predictions".format(name))
    plt.xlabel('Timestamps')
    plt.ylabel('Close')

    plt.plot(new_open_df['Actual Open'], linewidth=3, color='purple')
    plt.plot(new_open_df['Predicted Open'], linewidth=3, color='orange')

    plt.legend(['Test Close', 'Prediction'], loc='upper left')
    plt.savefig('{}_stock_price_prediction.png'.format(name), dpi=100)
    plt.show()"""

    #data = data.filter(['Close'])


    #n_data = data[len(data)+1 - 60:len(data+1),0]

    n_data = [data[len(data)+1 - 60:len(data+1),0]]

    #n_data = scaler.transform(n_data)

    #n_data = data[-60:]

    #dividers = scaler.scale_

    #open_div = 1 / dividers[0]
    #close_div = 1 / dividers[3]

    #print(n_data)"""

    """scaled_data = scaler.transform(n_data)

    test = []

    for i in range(60, scaled_data.shape[0]):
        test.append(scaled_data[i-60:i])

    test = np.array(test)"""

    """n_data = data.tail(1200)"""

    n_data = np.array(n_data)

    print(n_data.shape)
    #n_data = np.shape

    #n_data = np.expand_dims(n_data, axis = 1)

    n_data = np.reshape(n_data,(n_data.shape[0], n_data.shape[1], 1))

    predi = model.predict(n_data)

    predi_result = scaler.inverse_transform(predi)

    #predi_result = predi_result.flatten()

    #predi_result = scaler.inverse_transform(predi_result)

    print(predi_result)

    return predi_result

    #ndf = pd.DataFrame({"Prediction Results":predi_result}, index = [0])




    #close_pred = pd.DataFrame({"Close Predictions":predi_result[-60:]})

    #print(close_pred)"""

#import_data('SBIN')


