import os
import pandas as pd
from RLmodel import Model,get_state,Agent
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import copy
import pickle

class Feature_Extractor():
    def __init__(self):
        pass

    def SMA(self, data, windows):
        res = data.rolling(window = windows).mean()
        return res

    def EMA(self, data, windows):
        res = data.ewm(span = windows).mean()
        return res

    def MACD(self, data, long, short, windows):
        short_ = data.ewm(span = short).mean()
        long_ = data.ewm(span = long).mean()
        macd_ = short_ - long_
        res = macd_.ewm(span = windows).mean()
        return res

    def RSI(self, data, windows):
        delta = data.diff(1)
        up = delta.copy()
        down = delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        avg_up = up.rolling(window = windows).mean()
        avg_down = down.rolling(window = windows).mean()
        rs = avg_up/ abs(avg_down)
        rsi = 100. -(100./ (1. + rs))
        return rsi

    def bollinger_band(self, data, windows):
        sma = data.rolling(window = windows).mean()
        std = data.rolling(window = windows).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        return upper, lower

    def rsv(self, data, windows):
        min_ = data.rolling(window = windows).min()
        max_ = data.rolling(window = windows).max()
        res = (data - min_)/ (max_ - min_) * 100
        return res
def data_preprocessing(df, preprocessor):
  # input: DataFrame
  data = df.copy()
  data['pct_change'] = (data['Close'] - data['Close'].shift(1))/ data['Close'].shift(1)
  data['log_change'] = np.log(data['Close']/ data['Close'].shift(1))
  data['7ma'] = preprocessor.EMA(data['Close'], 7)
  data['14ma'] = preprocessor.EMA(data['Close'], 14)
  data['21ma'] = preprocessor.EMA(data['Close'], 21)
  data['7macd'] = preprocessor.MACD(data['Close'], 3, 11, 7)
  data['14macd'] = preprocessor.MACD(data['Close'], 7, 21, 14)
  data['7rsi'] = preprocessor.RSI(data['Close'], 7)
  data['14rsi'] = preprocessor.RSI(data['Close'], 14)
  data['21rsi'] = preprocessor.RSI(data['Close'], 21)
  data['7upper'], data['7lower'] = preprocessor.bollinger_band(data['Close'], 7)
  data['14upper'], data['14lower'] = preprocessor.bollinger_band(data['Close'], 14)
  data['21upper'], data['21lower'] = preprocessor.bollinger_band(data['Close'], 21)
  data['7rsv'] = preprocessor.rsv(data['Close'], 7)
  data['14rsv'] = preprocessor.rsv(data['Close'], 14)
  data['21rsv'] = preprocessor.rsv(data['Close'], 21)

  return data
preprocessor = Feature_Extractor()
data_training_path = os.path.join(os.getcwd(), 'DataTraining')

# Lấy danh sách các tệp tin .csv trong thư mục dataTraining
stocks = [os.path.join(data_training_path,i)  for i in os.listdir(data_training_path) if i.endswith('.csv')]

print(stocks)

# for file in stocks:
#     file_path = os.path.join(data_training_path, file)
    
#     # Đọc tệp CSV vào DataFrame
#     df = pd.read_csv(file_path)
    
#     # Loại bỏ các hàng có giá trị null
#     df_cleaned = df.dropna()
    
#     # Ghi lại tệp CSV đã được làm sạch
#     df_cleaned.to_csv(file_path, index=False)
    


input_size = 163
skip = 1
layer_size = 1024
output_size = 3
device = "cuda" if torch.cuda.is_available() else "cpu"


model = Model(input_size = input_size, layer_size = layer_size, output_size = output_size)
agent = None

for no, stock in enumerate(stocks):
    print('training stock %s'%(stock))
    df = pd.read_csv(stock)
    df = df[['Close']]
    df = data_preprocessing(df, preprocessor)
    df = df.dropna()
    real_trend = df['Close'].tolist()
    parameters = [df[cl].tolist() for cl in df.columns]
    minmax = MinMaxScaler(feature_range = (100, 200)).fit(np.array(parameters).T)
    scaled_parameters = minmax.transform(np.array(parameters).T).T.tolist()
    initial_money = 1000000


    if no == 0:
        agent = Agent(model = model,
                      timeseries = scaled_parameters,
                      skip = skip,
                      initial_money = initial_money,
                      real_trend = real_trend,
                      minmax = minmax)
    else:
        agent.change_data(timeseries = scaled_parameters,
                          skip = skip,
                          initial_money = initial_money,
                          real_trend = real_trend,
                          minmax = minmax)

    agent.fit(iterations = 100, checkpoint = 10)
    print()

copy_model = copy.deepcopy(agent.model)

with open('model.pkl', 'wb') as fopen:
    pickle.dump(copy_model, fopen)


ofile = open("model.pkl",'rb')
agent.model = pickle.load(ofile)