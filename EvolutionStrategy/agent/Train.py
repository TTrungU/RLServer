import argparse
import logging
import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import copy
import yaml
import pickle

import torch
import torch.nn as nn

from modules import Model, Agent, LSTM_Model
from preprocessing import Feature_Extractor, data_preprocessing
from utils import Config, Average


logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
#logging = logging.getLogger(__name__)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='path to config file')
    parser.add_argument('--data_path', type=str, required=True,
                        help='path to data path')
    parser.add_argument('--name', type=str, required=None)
    parser.add_argument('--initial_money', type=float, required=None)
    parser.add_argument('--checkpoint', type=str, required=None)

    opt = parser.parse_args()

    assert os.path.exists(opt.config), '%s does not exists!' % opt.config
    with open(opt.config) as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)
    opt.config = Config(config_dict)
    logging.info(opt)

    logging.info('Construct dataset.')
    df = pd.read_csv(opt.data_path)
    df = df[['Close']]
    df = data_preprocessing(df, Feature_Extractor)
    real_trend = df['Close'].tolist()
    parameters = [df[cl].tolist() for cl in df.columns]
    minmax = MinMaxScaler(feature_range = (100, 200)).fit(np.array(parameters).T)
    scaled_parameters = minmax.transform(np.array(parameters).T).T.tolist()

    if opt.initial_money:
        initial_money= opt.initial_money
    else:
        initial_money = np.max(parameters[0]) * 2

    #init model & agent
    logging.info('Construct agent.')
    model = Model(input_size = opt.config.input_size,
                  layer_size = opt.config.layer_size,
                  output_size = opt.config.output_size)
    if opt.checkpoint:
        ofile = open(opt.checkpoint)
        model = pickle.load(ofile)

    agent = Agent(model = model,
                  timeseries = scaled_parameters,
                  skip = opt.config.skip,
                  initial_money = initial_money,
                  real_trend = real_trend,
                  minmax = minmax,
                  window_size = opt.config.window_size)

    logging.info('Start training.')
    agent.fit(iterations = 100, checkpoint = 10)
    logging.info('Training complete.')
    #save scaler & model

    logging.info('Saving model')
    scalerfile = f'checkpoint/{opt.name}_scaler.pkl'
    pickle.dump(minmax, open(scalerfile, 'wb'))

    modelfile = f'checkpoint/{opt.name}_model.pkl'
    copy_model = copy.deepcopy(agent.model)
    pickle.dump(copy_model, open(modelfile, 'wb'))



if __name__ == '__main__':
    main()
