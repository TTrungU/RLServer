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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from dataset import TimeSeriesDataset, AlignCollate
from modules import VAE, Generator, Discriminator
from preprocessing import Feature_Extractor, data_preprocessing
from utils import Config, sliding_window_GAN_training

def root_mean_squared_error(y_actual,y_predicted):
   return mean_squared_error(y_actual, y_predicted, squared=False)


device = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
#logging = logging.getLogger(__name__)


def main(opt):
    logging.info('Construct dataset.')
    df = pd.read_csv(opt.data_path)
    df = df[['Close']]
    df = data_preprocessing(df, Feature_Extractor)

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

    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_x).float()), \
                              batch_size = opt.config.batch_size, shuffle = False)
    logging.info('Construct VAE model.')
    model = VAE([20, 256, 256, 256, 16], 16).to(device)

    logging.info('Start Training VAE.')

    num_epochs = 400

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.00004)

    hist = np.zeros(num_epochs)
    for epoch in range(num_epochs):
        total_loss = 0
        loss_ = []
        for (x, ) in train_loader:
            x = x.to(device)
            output, z, mu, logVar = model(x)
            kl_divergence = 0.5* torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
            #loss = F.binary_cross_entropy(output, x) + kl_divergence
            loss = F.mse_loss(output, x) + kl_divergence
            loss.backward()
            optimizer.step()
            loss_.append(loss.item())
        hist[epoch] = sum(loss_)
        print('[{}/{}] Loss:'.format(epoch+1, num_epochs), sum(loss_))

    logging.info("Save VAE model. ")
    torch.save(model.state_dict(), f"checkpoint/{opt.name}_VAE.pt")

    logging.info('Conduct data for GAN training')
    model.eval()
    _, VAE_train_x, train_x_mu, train_x_var = model(torch.from_numpy(train_x).float().to(device))
    _, VAE_test_x, test_x_mu, test_x_var = model(torch.from_numpy(test_x).float().to(device))

    del model, train_loader, optimizer

    train_x = np.concatenate((train_x, VAE_train_x.cpu().detach().numpy()), axis = 1)
    test_x = np.concatenate((test_x, VAE_test_x.cpu().detach().numpy()), axis = 1)

    train_x_slide, train_y_slide, train_y_gan = sliding_window_GAN_training(train_x, train_y, \
                                                                        window = opt.config.window_size)
    test_x_slide, test_y_slide, test_y_gan = sliding_window_GAN_training(test_x, test_y, \
                                                                     window = opt.config.window_size)
    print(f'train_x: {train_x_slide.shape} train_y: {train_y_slide.shape} train_y_gan: {train_y_gan.shape}')
    print(f'test_x: {test_x_slide.shape} test_y: {test_y_slide.shape} test_y_gan: {test_y_gan.shape}')

    trainDataloader = DataLoader(TensorDataset(train_x_slide, train_y_gan), \
                                 batch_size = opt.config.batch_size, shuffle = False)

    modelG = Generator(36).to(device)
    modelD = Discriminator(opt.config.window_size+1).to(device)

    optimizerG = torch.optim.Adam(modelG.parameters(), lr = 0.000115, \
                                  betas = (0.0, 0.9), weight_decay = 1e-3)
    optimizerD = torch.optim.Adam(modelD.parameters(), lr = 0.000115, \
                                  betas = (0.0, 0.9), weight_decay = 1e-3)

    num_epochs = 200
    critic_iterations = 5
    weight_clip = 0.01
    histG = np.zeros(num_epochs)
    histD = np.zeros(num_epochs)
    count = 0
    for epoch in range(num_epochs):
        loss_G = []
        loss_D = []
        for (x, y) in trainDataloader:
            x = x.to(device)
            y = y.to(device)

            fake_data = modelG(x)
            fake_data = torch.cat([y[:, :opt.config.window_size, :], fake_data.reshape(-1, 1, 1)], axis = 1)
            critic_real = modelD(y)
            critic_fake = modelD(fake_data)
            lossD = -(torch.mean(critic_real) - torch.mean(critic_fake))
            modelD.zero_grad()
            lossD.backward(retain_graph = True)
            optimizerD.step()

            output_fake = modelD(fake_data)
            lossG = -torch.mean(output_fake)
            modelG.zero_grad()
            lossG.backward()
            optimizerG.step()

            loss_D.append(lossD.item())
            loss_G.append(lossG.item())
        histG[epoch] = sum(loss_G)
        histD[epoch] = sum(loss_D)
        print(f'[{epoch+1}/{num_epochs}] LossD: {sum(loss_D)} LossG:{sum(loss_G)}')


    modelG.eval()
    pred_y_train = modelG(train_x_slide.to(device))
    pred_y_test = modelG(test_x_slide.to(device))

    y_train_true = y_scaler.inverse_transform(train_y_slide)
    y_train_pred = y_scaler.inverse_transform(pred_y_train.cpu().detach().numpy())

    y_test_true = y_scaler.inverse_transform(test_y_slide)
    y_test_pred = y_scaler.inverse_transform(pred_y_test.cpu().detach().numpy())

    rmse = root_mean_squared_error(y_train_true, y_train_pred)
    mse = mean_squared_error(y_train_true, y_train_pred)
    logging.info(f'Model get score on train set with MSE: {mse}, RMSE: {rmse}')

    rmse = root_mean_squared_error(y_test_true, y_test_pred)
    mse = mean_squared_error(y_test_true, y_test_pred)
    logging.info(f'Model get score on test set with MSE: {mse}, RMSE: {rmse}')

    torch.save(modelG.state_dict(), f"checkpoint/{opt.name}_modelG.pt")
    torch.save(modelD.state_dict(), f"checkpoint/{opt.name}_modelD.pt")

    with open(f'checkpoint/{opt.name}_GAN_xscaler.pkl', 'wb') as fopen:
        pickle.dump(x_scaler, fopen)
    with open(f'checkpoint/{opt.name}_GAN_yscaler.pkl', 'wb') as fopen:
        pickle.dump(y_scaler, fopen)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='path to config file')
    parser.add_argument('--data_path', type=str, required=True,
                        help='path to data path')
    parser.add_argument('--name', type=str, required=None)
    parser.add_argument('--checkpoint', type=str, required=None)

    opt = parser.parse_args()
    logging.info(opt)

    assert os.path.exists(opt.config), '%s does not exists!' % opt.config
    with open(opt.config) as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)
    logging.info(config_dict)

    opt.config = Config(config_dict)

    main(opt)
