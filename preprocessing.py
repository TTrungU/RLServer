import pandas as pd
import numpy as np

class Feature_Extractor():
    def __init__(self):
        pass

    @staticmethod
    def SMA(data, windows):
        res = data.rolling(window = windows).mean()
        return res

    @staticmethod
    def EMA(data, windows):
        res = data.ewm(span = windows).mean()
        return res

    @staticmethod
    def MACD(data, long, short, windows):
        short_ = data.ewm(span = short).mean()
        long_ = data.ewm(span = long).mean()
        macd_ = short_ - long_
        res = macd_.ewm(span = windows).mean()
        return res

    @staticmethod
    def RSI(data, windows):
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

    @staticmethod
    def bollinger_band(data, windows):
        sma = data.rolling(window = windows).mean()
        std = data.rolling(window = windows).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        return upper, lower

    @staticmethod
    def rsv(data, windows):
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
  data = data.dropna()

  return data

