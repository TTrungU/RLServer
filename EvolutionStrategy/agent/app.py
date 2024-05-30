from flask import Flask, request, jsonify
import numpy as np
import pickle
import json
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from datetime import datetime
from pymongo import MongoClient
import config
from bson import json_util
from preprocessing import Feature_Extractor, data_preprocessing
from modules import Agent,Model
from LSTMPredict import LSTMPredict, GANPredict
from modules import LSTM_Model, VAE, Generator
import torch
from DCAStrategy import DCAAgent
from LSStrategy import LSSAgent

app = Flask(__name__)


client = MongoClient(config.MONGO_URI) # your connection string
db = client["StockDB"]
trading_collection = db["Trading"]
stock_collection = db["StockData"]

device = "cuda" if torch.cuda.is_available() else "cpu"
@app.route('/', methods = ['GET'])
def hello():
    return jsonify({'status': 'OK'})
@app.route('/reset_agent', methods = ['POST'])
def reset_agent():
    data = request.json
    df = pd.read_csv('TWTR.csv')
    real_trend = df['Close'].tolist()
    parameters = [df['Close'].tolist(), df['Volume'].tolist()]
    minmax = MinMaxScaler(feature_range = (100, 200)).fit(np.array(parameters).T)
    scaled_parameters = minmax.transform(np.array(parameters).T).T.tolist()
    initial_money = data.money

    agent.change_data(timeseries = scaled_parameters,
                  skip = skip,
                  initial_money = initial_money,
                  real_trend = real_trend,
                  minmax = minmax)

    return jsonify(True)

@app.route('/inventory', methods = ['GET'])
def inventory():
    return jsonify(agent._inventory)


@app.route('/queue', methods = ['GET'])
def queue():
    return jsonify(agent._queue)


@app.route('/balance', methods = ['GET'])
def balance():
    return jsonify(agent._capital)


# @app.route('/trade', methods = ['GET'])
# def trade():
#     data = json.loads(request.args.get('data'))
#     return jsonify(agent.trade(data))

@app.route('/trade', methods=['POST'])
def trade():
    data = request.json
    result = agent.trade([data.get('close', 0), data.get('volume', 0)])
    return jsonify(result)

@app.route('/LSTMPredict',methods = ['GET'])
def LSTM_Predict():
    # data = request.json
    # symbol = data.get('Symbol')
    symbol = 'SINA'
    data = pd.read_csv('SINA.csv')
    model = LSTM_Model(input_size = 20,
                                output_size = 1)
    model.to(device)
    model.load_state_dict(torch.load(f"checkpoint/{symbol}_forecast_model.pt"))
    x_scaler = pickle.load(open(f"checkpoint/{symbol}_LSTM_xscaler.pkl", 'rb'))
    y_scaler = pickle.load(open(f"checkpoint/{symbol}_LSTM_yscaler.pkl", 'rb'))
    result = LSTMPredict(data,x_scaler,y_scaler,model)
    print(result)
    return jsonify(result.to_json())

@app.route('/GAN_Predict',methods = ['GET'])
def GAN_Predict():
    # data = request.json
    # symbol = data.get('Symbol')
    symbol = 'ACB'
    data = pd.read_csv('DataTraining/ACB.csv')
    
    model_VAE = VAE([20, 256, 256, 256, 16], 16).to(device)
    modelG = Generator(36).to(device)

    model_VAE.load_state_dict(torch.load(f"checkpoint/{symbol}_VAE.pt"))
    modelG.load_state_dict(torch.load(f"checkpoint/{symbol}_modelG.pt"))
    x_scaler = pickle.load(open(f"checkpoint/{symbol}_GAN_xscaler.pkl", 'rb'))
    y_scaler = pickle.load(open(f"checkpoint/{symbol}_GAN_yscaler.pkl", 'rb'))

    result = GANPredict(data, x_scaler, y_scaler, model_VAE, modelG)
    print(result)
    return jsonify(result.to_json())

@app.route('/trade_range', methods=['POST'])
def trade_range():
    data = request.json
    from_date = data.get('from_date')
    to_date = data.get('to_date')
    symbol = data.get('symbol')
    money = data.get('init_money')
    strategy = data.get('strategy')

    document = stock_collection.find_one({'Stockinfor.Symbol': symbol}, {'stockData.Date': 1, 'stockData.Close': 1, 'stockData.High': 1, 'stockData.Low': 1, '_id': 0})

# Extracting stockData from the document
    if document:
        stock_data = document.get('stockData', [])
        df = pd.DataFrame(stock_data)
        print("DataFrame from MongoDB:", df)
    else:
        return jsonify({"message": "Not Found"}),404
    df_init = df
    df['Date'] = pd.to_datetime(df['Date'])
    df_init = df_init[['Close']]  
    df_init = data_preprocessing(df_init, Feature_Extractor)
    real_trend = df_init['Close'].tolist()
    parameters = [df_init[cl].tolist() for cl in df_init.columns]
    minmax = pickle.load(open(f"checkpoint/{symbol}_scaler.pkl", 'rb'))
    scaled_parameters = minmax.transform(np.array(parameters).T).T.tolist()
    # initial_money = np.max(parameters[0]) * 2
    skip = 1
    if(strategy == 'DCA'):
        with open(f"checkpoint/{symbol}_DCAmodel.pkl", 'rb') as fopen:
            model = pickle.load(fopen)
        agent = DCAAgent(model = model,
                timeseries = scaled_parameters,
                skip = skip,
                initial_money = money,
                real_trend = real_trend,
                minmax = minmax,
                window_size= 10)
        print(strategy)
    elif(strategy == 'LSS'):
        with open(f"checkpoint/{symbol}_LSSmodel.pkl", 'rb') as fopen:
            model = pickle.load(fopen)
        agent = LSSAgent(model = model,
                timeseries = scaled_parameters,
                skip = skip,
                initial_money = money,
                real_trend = real_trend,
                minmax = minmax,
                window_size= 10)    
    else:
        with open(f"checkpoint/{symbol}_model.pkl", 'rb') as fopen:
            model = pickle.load(fopen)
        agent = Agent(model = model,
                timeseries = scaled_parameters,
                skip = skip,
                initial_money = money,
                real_trend = real_trend,
                minmax = minmax,
                window_size= 10)
    
    from_date = pd.to_datetime(from_date)
    to_date = pd.to_datetime(to_date)

    df = df[['Date', 'Close']]
    #Preprocess Dataframe
    df = data_preprocessing(df, Feature_Extractor)

    selected_data = df.loc[(df['Date'] >= from_date) & (df['Date'] <= to_date),:]
    print(selected_data.columns)
    if selected_data.empty:
        return jsonify({'message': 'No data available for the selected period.'}), 400

    data_list = selected_data.values.tolist()
    
    trade_results = []
    for row in data_list:
        #ensure first column is Date
        date = row[0]
        value = row[1:]
        result = agent.trade(value, date = date)
        trade_results.append(result)

    trading_collection.insert_one({
        "UserId": data.get('UserId'),
        "StockSymbol": data.get('Symbol'),
        "HistoryTrading":trade_results
    })
    return jsonify(trade_results)


@app.route('/trade_history', methods = ['GET'])
def trade_history():
    userId = request.args.get('UserId')
    symbol = request.args.get('StockSymbol')

    if isinstance(userId, str):
            userId = int(userId)

    if not userId or not symbol:
        return jsonify({'error': 'UserId and Symbol parameters are required'}), 400
    print("userid,symbol",userId,symbol)

    document  =  trading_collection.find_one({
        "UserId": userId,
        "StockSymbol": symbol
    })
    
    return jsonify(json_util.dumps(document))


@app.route('/all_trade_history',methods = ['GET'])
def all_trade_history():
    all_documents = list(trading_collection.find())
   
    for doc in all_documents:
        doc['_id'] = str(doc['_id'])
    return jsonify(json_util.dumps(all_documents))

@app.route('/reset', methods = ['GET'])
def reset():
    money = json.loads(request.args.get('money'))
    agent.reset_capital(money)
    return jsonify(True)


if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 8005)
