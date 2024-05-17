from flask import Flask, request, jsonify
import numpy as np
import pickle
import json
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from datetime import datetime
from RLmodel import Agent,skip,Model
from pymongo import MongoClient
import config
from bson import json_util
app = Flask(__name__)

client = MongoClient(config.MONGO_URI) # your connection string
db = client["StockDB"]
trading_collection = db["Trading"]
stock_collection = db["Stock"]

with open('model.pkl', 'rb') as fopen:
    model = pickle.load(fopen)

df = pd.read_csv('TSLA.csv')
real_trend = df['Close'].tolist()
parameters = [df['Close'].tolist(), df['Volume'].tolist()]
minmax = MinMaxScaler(feature_range = (100, 200)).fit(np.array(parameters).T)
scaled_parameters = minmax.transform(np.array(parameters).T).T.tolist()
initial_money = np.max(parameters[0]) * 2

agent = Agent(model = model,
              timeseries = scaled_parameters,
              skip = skip,
              initial_money = initial_money,
              real_trend = real_trend,
              minmax = minmax)

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


@app.route('/trade_range', methods=['POST'])
def trade_range():
    data = request.json
    from_date = data.get('from_date')
    to_date = data.get('to_date')   
   
    df['Date'] = pd.to_datetime(df['Date'])
    
    from_date = pd.to_datetime(from_date)
    to_date = pd.to_datetime(to_date)
     
    selected_data = df.loc[(df['Date'] >= from_date) & (df['Date'] <= to_date), ['Date','Close']] 
 
    if selected_data.empty:
        return jsonify({'message': 'No data available for the selected period.'}), 400    
  
    data_list = selected_data.values.tolist()   
  
    trade_results = []
    for date, close in data_list:
        result = agent.trade([date,close])
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
