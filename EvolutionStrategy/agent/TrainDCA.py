
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import argparse
import logging
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import copy
import yaml
import pickle
from modules import Model
from DCAStrategy import DCAAgent
from sklearn.preprocessing import MinMaxScaler
from preprocessing import Feature_Extractor, data_preprocessing
from utils import Config


device = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
#logging = logging.getLogger(__name__)


# Training function
def Trainer(forecast_model, train_loader, val_loader, criterion, optimizer, train_loss_avg, eval_loss_avg):
  forecast_model.train()
  for batch in train_loader:

      input_tensor, gt_tensors = batch

      inputs = input_tensor.to(device)
      gts = gt_tensors.to(device)
      preds = forecast_model(inputs.contiguous())

      loss = criterion(preds.reshape(-1), gts.reshape(-1))

      # optimize
      forecast_model.zero_grad(set_to_none=True)
      loss.backward()
      optimizer.step()
      train_loss_avg.add(loss)

  lr = optimizer.param_groups[0]["lr"]
  epoch_loss = train_loss_avg.val()

  train_loss_avg.reset()

  val_loss = Validation(forecast_model, criterion, val_loader, eval_loss_avg)

  return epoch_loss, val_loss, lr

# Validating Function
def Validation(forecast_model, criterion, val_loader, eval_loss_avg):
  forecast_model.eval()
  with torch.no_grad():
    for batch in val_loader:

        input_tensor, gt_tensors = batch

        inputs = input_tensor.to(device)
        gts = gt_tensors.to(device)
        preds = forecast_model(inputs.contiguous())
        loss = criterion(preds, gts)

        eval_loss_avg.add(loss)

  val_loss = eval_loss_avg.val()
  eval_loss_avg.reset()
  return val_loss

# Evaluating Function
def Evaluation(forecast_model, dataloader):
  y_pred = []
  y_true = []

  forecast_model.eval()
  with torch.no_grad():
    for batch in (dataloader):

      input_tensor, gt_tensors = batch

      inputs = input_tensor.to(device)
      gts = gt_tensors.to(device)
      preds = forecast_model(inputs.contiguous())
      y_true.append(gt_tensors.numpy())
      y_pred.append(preds.cpu().numpy())

  return np.concatenate(y_true,axis=0), np.concatenate(y_pred,axis=0)

def main(opt):
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
        initial_money = np.max(parameters[0]) * 20

    #init model & agent
    logging.info('Construct agent.')
    model = Model(input_size = opt.config.input_size,
                  layer_size = opt.config.layer_size,
                  output_size = opt.config.output_size)
    if opt.checkpoint:
        ofile = open(opt.checkpoint)
        model = pickle.load(ofile)

    agent = DCAAgent(model = model,
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

    modelfile = f'checkpoint/{opt.name}_DCAmodel.pkl'
    copy_model = copy.deepcopy(agent.model)
    pickle.dump(copy_model, open(modelfile, 'wb'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='path to config file')
    parser.add_argument('--data_path', type=str, required=True,
                        help='path to data path')
    parser.add_argument('--name', type=str, required=None)
    parser.add_argument('--initial_money', type=float, required=None)
    parser.add_argument('--checkpoint', type=str, required=None)

    opt = parser.parse_args()
    logging.info(opt)

    assert os.path.exists(opt.config), '%s does not exists!' % opt.config
    with open(opt.config) as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)
    logging.info(config_dict)

    opt.config = Config(config_dict)

    main(opt)
