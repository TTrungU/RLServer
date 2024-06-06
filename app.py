from flask import Flask, request, jsonify
import numpy as np
import pickle
import json
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from datetime import timedelta
from pymongo import MongoClient
import config
from bson import json_util
from preprocessing import Feature_Extractor, data_preprocessing
from modules import Agent,Model
from LSTMPredict import LSTMPredict, GANPredict,slide_window
from modules import LSTM_Model, VAE, Generator
import torch
from DCAStrategy import DCAAgent
from LSStrategy import LSSAgent
from AnomalyDetection import Detection
app = Flask(__name__)


client = MongoClient(config.MONGO_URI) # your connection string
db = client["StockDB"]
trading_collection = db["Trading"]
stock_collection = db["StockData"]

device = "cuda" if torch.cuda.is_available() else "cpu"
@app.route('/', methods = ['GET'])
def hello():
    return jsonify({'status': 'OK'})
# @app.route('/reset_agent', methods = ['POST'])
# def reset_agent():
#     data = request.json
#     df = pd.read_csv('TWTR.csv')
#     real_trend = df['Close'].tolist()
#     parameters = [df['Close'].tolist(), df['Volume'].tolist()]
#     minmax = MinMaxScaler(feature_range = (100, 200)).fit(np.array(parameters).T)
#     scaled_parameters = minmax.transform(np.array(parameters).T).T.tolist()
#     initial_money = data.money

#     agent.change_data(timeseries = scaled_parameters,
#                   skip = skip,
#                   initial_money = initial_money,
#                   real_trend = real_trend,
#                   minmax = minmax)

#     return jsonify(True)

# @app.route('/inventory', methods = ['GET'])
# def inventory():
#     return jsonify(agent._inventory)


# @app.route('/queue', methods = ['GET'])
# def queue():
#     return jsonify(agent._queue)


# @app.route('/balance', methods = ['GET'])
# def balance():
#     return jsonify(agent._capital)


# @app.route('/trade', methods = ['GET'])
# def trade():
#     data = json.loads(request.args.get('data'))
#     return jsonify(agent.trade(data))

# @app.route('/trade', methods=['POST'])
# def trade():
#     data = request.json
#     result = agent.trade([data.get('close'))
#     return jsonify(result)

@app.route('/LSTMPredict',methods = ['GET'])
def LSTM_Predict():
    symbol = request.args.get('Symbol')
    data = pd.read_csv(f'DataTraining/{symbol}.csv')
    model = LSTM_Model(input_size = 20,
                                output_size = 1)
    model.to(device)
    model.load_state_dict(torch.load(f"checkpoint/{symbol}_forecast_model.pt"))
    x_scaler = pickle.load(open(f"checkpoint/{symbol}_LSTM_xscaler.pkl", 'rb'))
    y_scaler = pickle.load(open(f"checkpoint/{symbol}_LSTM_yscaler.pkl", 'rb'))
    result = LSTMPredict(data,x_scaler,y_scaler,model)
    result = Detection(result)
    result['Date'] = result['Date'].dt.strftime("%Y-%m-%d")
    result = result.tail(29)
    return jsonify(result.to_dict(orient='records'))

@app.route('/GANPredict',methods = ['GET'])
def GAN_Predict():
    symbol = request.args.get('Symbol')
    data = pd.read_csv(f'DataTraining/{symbol}.csv')
    
    model_VAE = VAE([20, 256, 256, 256, 16], 16).to(device)
    modelG = Generator(36).to(device)

    model_VAE.load_state_dict(torch.load(f"checkpoint/{symbol}_VAE.pt"))
    modelG.load_state_dict(torch.load(f"checkpoint/{symbol}_modelG.pt"))
    x_scaler = pickle.load(open(f"checkpoint/{symbol}_GAN_xscaler.pkl", 'rb'))
    y_scaler = pickle.load(open(f"checkpoint/{symbol}_GAN_yscaler.pkl", 'rb'))

    result = GANPredict(data, x_scaler, y_scaler, model_VAE, modelG)
    result = Detection(result)
    result['Date'] = result['Date'].dt.strftime("%Y-%m-%d")
    result = result.tail(29)
    return jsonify(result.to_dict(orient='records'))

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
 
    
    # initial_money = np.max(parameters[0]) * 2
    skip = 1
    if(strategy == 'DCA'):
        minmax = pickle.load(open(f"checkpoint/{symbol}_DCAscaler.pkl", 'rb'))
        scaled_parameters = minmax.transform(np.array(parameters).T).T.tolist()
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
        minmax = pickle.load(open(f"checkpoint/{symbol}_LSSscaler.pkl", 'rb'))
        scaled_parameters = minmax.transform(np.array(parameters).T).T.tolist()
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
        minmax = pickle.load(open(f"checkpoint/{symbol}_scaler.pkl", 'rb'))
        scaled_parameters = minmax.transform(np.array(parameters).T).T.tolist()
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


@app.route('/LSTMTradingPredict',methods = ['GET'])
def LSTMTradingPredict():
    symbol = request.args.get('Symbol')
    data = pd.read_csv(f'DataTraining/{symbol}.csv')

    df_feedback = data.copy().reset_index(drop=True)
    df_feedback['Date'] = pd.to_datetime(df_feedback['Date'])
    df_feedback['forecast'] = np.nan
    df_feedback['signal'] = np.nan

    LSTMmodel = LSTM_Model(input_size = 20,
                                output_size = 1)
    LSTMmodel.to(device)
    LSTMmodel.load_state_dict(torch.load(f"checkpoint/{symbol}_forecast_model.pt"))
    x_scaler = pickle.load(open(f"checkpoint/{symbol}_LSTM_xscaler.pkl", 'rb'))
    y_scaler = pickle.load(open(f"checkpoint/{symbol}_LSTM_yscaler.pkl", 'rb'))
    # result = LSTMPredict(data,x_scaler,y_scaler,LSTMmodel)


    data['Date'] = pd.to_datetime(data['Date'])
    data = data[['Close']]  
    data = data_preprocessing(data, Feature_Extractor)
   
    real_trend = data['Close'].tolist()
    parameters = [data[cl].tolist() for cl in data.columns]
    minmax = pickle.load(open(f"checkpoint/{symbol}_scaler.pkl", 'rb'))
    scaled_parameters = minmax.transform(np.array(parameters).T).T.tolist()
    initial_money = np.max(parameters[0]) * 5
    with open(f"checkpoint/{symbol}_model.pkl", 'rb') as fopen:
            model = pickle.load(fopen)
    agent = Agent(model = model,
            timeseries = scaled_parameters,
            skip = 1,
            initial_money = initial_money,
            real_trend = real_trend,
            minmax = minmax,
            window_size= 10)
    window_size = 5
    prev_window_data = np.array(scaled_parameters)[:,-window_size:].T
    agent._queue = [prev_window_data[i] for i in range(window_size)]
    states_sell = []
    states_buy = []
        
    future_predicted = list()
    data_tmp = data[['Close']][-60:]
    LSTMmodel.eval()
    date_record = df_feedback[~df_feedback['Close'].isna()]
    current_date = date_record['Date'][len(date_record)-1]

    current_index = len(date_record)-1
    for i in range(60):
        data_tmp_preprocessed = data_preprocessing(data_tmp, Feature_Extractor)
        if i >0:
            response = agent.trade(data_tmp_preprocessed.values[-1].tolist())
            print(i-1, response)
            if response['action'] == 2:
                states_sell.append(i-1)
                new_row.update({"signal" : "sell"})
            elif response['action'] == 1:
                states_buy.append(i-1)
                new_row.update({"signal" : "buy"})
            else:
                new_row.update({"signal" : "nothing"})

            current_index +=1
            df_feedback.loc[current_index] = new_row
        x_future = x_scaler.transform(data_tmp_preprocessed.values)
        x_future = slide_window(x_future, window_size  = 10)[-1]
        prediction = LSTMmodel(torch.from_numpy(x_future).unsqueeze(0).float().to(device))

        prediction_inv = y_scaler.inverse_transform(prediction.cpu().detach().numpy())

        data_tmp.loc[len(data_tmp)] = {"Close": prediction_inv[0][0]}
        new_row = {"Date" : df_feedback['Date'][current_index] + timedelta(days=1),\
                                            "forecast": prediction_inv[0][0]}

        future_predicted.append(prediction_inv[0][0])

    df_feedback['Date'] = df_feedback['Date'].dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]
    # result = result.tail(29)
    return jsonify(df_feedback.to_dict(orient='records'))
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


# @app.route('/all_trade_history',methods = ['GET'])
# def all_trade_history():
#     all_documents = list(trading_collection.find())
   
#     for doc in all_documents:
#         doc['_id'] = str(doc['_id'])
#     return jsonify(json_util.dumps(all_documents))

# @app.route('/reset', methods = ['GET'])
# def reset():

#     money = json.loads(request.args.get('money'))
#     agent.reset_capital(money)
#     return jsonify(True)
print("hello")

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 8000)
