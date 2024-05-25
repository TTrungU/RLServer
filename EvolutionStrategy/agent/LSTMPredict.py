import random
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from Train import data_preprocessing,device
from modules import LSTM_Model
from preprocessing import Feature_Extractor
# data
def slide_window(data, window_size):
    X = []
    for i in range(len(data) - window_size + 1):
        X.append(data[i:i+window_size])
    return np.array(X)

def LSTMPredict(data,x_scaler,y_scaler,model):
    df_feedback = data.copy().reset_index(drop=True)
    df_feedback['Date'] = pd.to_datetime(df_feedback['Date'])
    df_feedback['forecast'] = np.nan
    df_feedback['signal'] = np.nan

    date_record = df_feedback[~df_feedback['Close'].isna()]
    # init a sample update row
    update_row= df_feedback.iloc[len(date_record) -1]
    update_row['Date']= update_row['Date'] + timedelta(days=1)

    update_index = len(date_record)
    #upload new record from realtime to database
    df_feedback['Open'][update_index] = update_row['Open']
    df_feedback['High'][update_index] = update_row['High']
    df_feedback['Low'][update_index] = update_row['Low']
    df_feedback['Close'][update_index] = update_row['Close']
    df_feedback['Adj Close'][update_index] = update_row['Adj Close']
    df_feedback['Volume'][update_index] = update_row['Volume']



    #adapt new real time data for agent
    data_record = df_feedback[~df_feedback['Close'].isna()]
    current_date = data_record['Date'][len(data_record)-1]

    data_record_Close = date_record[['Close']]

    data_record_Close = data_preprocessing(data_record_Close, Feature_Extractor)


    future_predicted = list()
    data_tmp = date_record[['Close']][-60:]
    model.eval()

    current_index = len(data_record)-1

    with torch.no_grad():
        for i in range(30):
            data_tmp_preprocessed = data_preprocessing(data_tmp, Feature_Extractor)

            if i == 0:
                new_row = df_feedback.iloc[current_index]

            if i>0:
                current_index +=1
            print(current_index, new_row)
            df_feedback.loc[current_index] = new_row

            x_future = x_scaler.transform(data_tmp_preprocessed.values)
            x_future = slide_window(x_future, window_size  = 10)[-1]
            prediction = model(torch.from_numpy(x_future).unsqueeze(0).float().to(device))

            prediction_inv = y_scaler.inverse_transform(prediction.cpu().detach().numpy())

            data_tmp.loc[len(data_tmp)] = {"Close": prediction_inv[0][0]}
            new_row = {"Date" : df_feedback['Date'][current_index] + timedelta(days=1),\
                                                "forecast": prediction_inv[0][0]}

            future_predicted.append(prediction_inv[0][0])
    return df_feedback


