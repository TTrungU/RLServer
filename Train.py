import argparse
import logging
import os

from tqdm import tqdm
import pandas as pd
import numpy as np
import copy
import yaml
import pickle

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dataset import TimeSeriesDataset, AlignCollate
from modules import Model, Agent, LSTM_Model
from preprocessing import Feature_Extractor, data_preprocessing
from utils import Config, Averager, sliding_window_training

def root_mean_squared_error(y_actual,y_predicted):
   return mean_squared_error(y_actual, y_predicted, squared=False)


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
        initial_money = np.max(parameters[0]) * 5

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
    agent.fit(iterations = 200, checkpoint = 10)
    logging.info('Training complete.')
    #save scaler & model

    logging.info('Saving model')
    scalerfile = f'checkpoint/{opt.name}_scaler.pkl'
    pickle.dump(minmax, open(scalerfile, 'wb'))

    modelfile = f'checkpoint/{opt.name}_model.pkl'
    copy_model = copy.deepcopy(agent.model)
    pickle.dump(copy_model, open(modelfile, 'wb'))

    logging.info('Conduct Data for Forecasting')
    ###
    df['y'] = df['Close']
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    split = int(df.shape[0]* opt.config.train_ratio)

    train_x, test_x = x[: split, :],  x[split:, :]
    train_y, test_y = y[: split, ], y[split: , ]

    print(f'trainX: {train_x.shape} trainY: {train_y.shape}')
    print(f'testX: {test_x.shape} testY: {test_y.shape}')

    x_scaler = MinMaxScaler(feature_range = (0, 10))
    y_scaler = MinMaxScaler(feature_range = (0, 10))

    train_x = x_scaler.fit_transform(train_x)
    test_x = x_scaler.transform(test_x)

    train_y = y_scaler.fit_transform(train_y.reshape(-1, 1))
    test_y = y_scaler.transform(test_y.reshape(-1, 1))

    train_x_slide, train_y_slide = sliding_window_training(train_x, train_y,
                                                           window = opt.config.window_size)
    test_x_slide, test_y_slide = sliding_window_training(test_x, test_y,
                                                         window = opt.config.window_size)

    train_set = TimeSeriesDataset(train_x_slide, train_y_slide)
    val_set = TimeSeriesDataset(test_x_slide, test_y_slide)

    Custom_AlignCollate = AlignCollate()

    train_loader = DataLoader(train_set, batch_size= opt.config.batch_size,
                              shuffle= False,
                              collate_fn= Custom_AlignCollate,
                              num_workers= 2, drop_last= False)
    val_loader = DataLoader(val_set, batch_size=opt.config.batch_size,
                            shuffle= False,
                            collate_fn= Custom_AlignCollate,
                            num_workers= 2, drop_last= False)

    logging.info('Conduct Model Forecasting')
    forecast_model = LSTM_Model(input_size = opt.config.num_parameters,
                                output_size = 1)
    forecast_model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(params = forecast_model.parameters(),
                                  lr= opt.config.lr,
                                  weight_decay = opt.config.weight_decay)

    train_loss_avg = Averager()
    eval_loss_avg = Averager()
    best_score = float('inf')

    for epoch in tqdm(range(opt.config.num_epoch)):
        #Training and Validation
        epoch_loss, val_loss, lr = Trainer(forecast_model, train_loader, val_loader, criterion, optimizer, train_loss_avg, eval_loss_avg)

        epoch_log = f"\nEpoch: {epoch + 1}/{opt.config.num_epoch}: "
        epoch_log += f"Train_loss: {epoch_loss:0.5f}, Val_loss: {val_loss:0.5f}, Current_lr: {lr:0.7f} \n"

        if val_loss <= best_score:
            torch.save(
                forecast_model.state_dict(),
                f"checkpoint/{opt.name}_forecast_model.pt"
            )
            best_score = val_loss

            logging.info(f"Saving model at epoch {epoch} with Validation Loss {val_loss}")
            epoch_log += "-"*16

        print(epoch_log)

    logging.info('Login best forecast model for evaluation')
    forecast_model.load_state_dict(torch.load(f"checkpoint/{opt.name}_forecast_model.pt"))

    #Evaluate model
    y_true, y_pred = Evaluation(forecast_model, val_loader)
    gt = y_scaler.inverse_transform(y_true)
    prd = y_scaler.inverse_transform(y_pred)
    
    rmse = root_mean_squared_error(gt, prd)
    mse = mean_squared_error(gt, prd)
    logging.info(f'Best model get score with MSE: {mse}, RMSE: {rmse}')
    with open(f'checkpoint/{opt.name}_LSTM_xscaler.pkl', 'wb') as fopen:
        pickle.dump(x_scaler, fopen)
    with open(f'checkpoint/{opt.name}_LSTM_yscaler.pkl', 'wb') as fopen:
        pickle.dump(y_scaler, fopen)    

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