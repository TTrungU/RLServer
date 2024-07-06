import numpy as np
import time
import math
from datetime import timedelta
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from utils import softmax, get_state


class Deep_Evolution_Strategy:

    inputs = None

    def __init__(
        self, weights, reward_function, population_size, sigma, learning_rate
    ):
        self.weights = weights
        self.reward_function = reward_function
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate

    def _get_weight_from_population(self, weights, population):
        weights_population = []
        for index, i in enumerate(population):
            jittered = self.sigma * i
            weights_population.append(weights[index] + jittered)
        return weights_population

    def get_weights(self):
        return self.weights

    def train(self, epoch = 200, print_every = 1):
        lasttime = time.time()
        for i in range(epoch):
            population = []
            rewards = np.zeros(self.population_size)
            for k in range(self.population_size):
                x = []
                for w in self.weights:
                    x.append(np.random.randn(*w.shape))
                population.append(x)
            for k in range(self.population_size):
                weights_population = self._get_weight_from_population(
                    self.weights, population[k]
                )
                rewards[k] = self.reward_function(weights_population)
            rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-7)
            for index, w in enumerate(self.weights):
                A = np.array([p[index] for p in population])
                self.weights[index] = (
                    w
                    + self.learning_rate
                    / (self.population_size * self.sigma)
                    * np.dot(A.T, rewards).T
                )
            if (i + 1) % print_every == 0:
                print(
                    'iter %d. reward: %f'
                    % (i + 1, self.reward_function(self.weights))
                )
        print('time taken to train:', time.time() - lasttime, 'seconds')

class Model:
    def __init__(self, input_size, layer_size, output_size):
        self.weights = [
            np.random.rand(input_size, layer_size)
            * np.sqrt(1 / (input_size + layer_size)),
            np.random.rand(layer_size, output_size)
            * np.sqrt(1 / (layer_size + output_size)),
            np.zeros((1, layer_size)),
            np.zeros((1, output_size)),
        ]

    def predict(self, inputs):
        feed = np.dot(inputs, self.weights[0]) + self.weights[-2]
        decision = np.dot(feed, self.weights[1]) + self.weights[-1]
        return decision

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights


class Agent:

    POPULATION_SIZE = 15
    SIGMA = 0.1
    LEARNING_RATE = 0.03

    def __init__(self, model, timeseries, skip, initial_money, real_trend, minmax, window_size):
        self.model = model
        self.timeseries = timeseries
        self.skip = skip
        self.real_trend = real_trend
        self.initial_money = initial_money
        self.es = Deep_Evolution_Strategy(
            self.model.get_weights(),
            self.get_reward,
            self.POPULATION_SIZE,
            self.SIGMA,
            self.LEARNING_RATE,
        )
        self.minmax = minmax
        self.window_size = window_size
        self._initiate()

    def _initiate(self):
        # i assume first index is the close value
        self.trend = self.timeseries[0]
        self.num_features = len(self.timeseries)
        self._mean = np.mean(self.trend)
        self._std = np.std(self.trend)
        self._inventory = []
        self._capital = self.initial_money
        self._queue = []
        self._scaled_capital = self.minmax.transform([[self._capital] * self.num_features])[0, 0]
        self._totalbuy = []
        self._totalsell = []
  

    def reset_capital(self, capital):
        if capital:
            self._capital = capital
        self._scaled_capital = self.minmax.transform([[self._capital] * self.num_features])[0, 0]
        self._queue = []
        self._inventory = []
        self._totalbuy = []
        self._totalsell = []
    

    def trade(self, data, date = None):
        window_size = 10
        scaled_data = self.minmax.transform([data])[0]
        real_close = data[0]
        close = scaled_data[0]
        if len(self._queue) >= window_size:
            self._queue.pop(0)
        self._queue.append(scaled_data)
        if len(self._queue) < window_size:
            return {
                'status': 'Do nothing',
                'action': 0,
                'close': real_close,
                'balance': self._capital,
                'timestamp': str(datetime.now()),
                'date': date.strftime("%Y-%m-%d"),
            }
        state = self.get_state(
            window_size - 1,
            self._inventory,
            self._scaled_capital,
            timeseries = np.array(self._queue).T.tolist(),
        )
        action, prob = self.act_softmax(state)
        print(prob)
        if action == 1 and self._scaled_capital >= close:
            self._inventory.append(close)
            self._scaled_capital -= close
            self._capital -= real_close
            total_units = len(self._inventory)
            total = total_units * real_close+ self._capital  
            return {
                'status': 'Buy 1 unit, cost %.2f' % (round(real_close,2)),
                'action': 1,
                'balance': self._capital,
                'close': real_close,
                'timestamp': str(datetime.now()),
                'total': total,
                'date': date.strftime("%Y-%m-%d"),
            }
        elif action == 2 and len(self._inventory):
            bought_price = self._inventory.pop(0)
            self._scaled_capital += close
            self._capital += real_close
            scaled_bought_price = self.minmax.inverse_transform(
                [[bought_price] * self.num_features]
            )[0, 0]
            self._totalbuy.append(scaled_bought_price)
            self._totalsell.append(real_close)
            totalBuy = sum(self._totalbuy)
            totalSell =sum(self._totalsell)
            totalinvest = ( totalSell-totalBuy )/totalBuy *100
            total_units = len(self._inventory)
            total = total_units * real_close+ self._capital 

            try:
                invest = (
                    (real_close - scaled_bought_price) / scaled_bought_price
                ) * 100
            except:
                invest = 0
            return {
                'status': 'Sell 1 unit, price %.2f' % (round(real_close,2)),
                'investment': invest,
                'total_investment':totalinvest,
                'all_bought': totalBuy,
                'all_sold': totalSell,
                'gain': real_close - scaled_bought_price,
                'balance': self._capital,
                'action': 2,
                'close': real_close,
                'total': total,
                'timestamp': str(datetime.now()),
                'date': date.strftime("%Y-%m-%d"),
            }
        else:
            total_units = len(self._inventory)
            total = total_units * real_close+ self._capital  
            return {
                'status': 'Do nothing',
                'date':date.strftime("%Y-%m-%d"),
                'close':real_close,
                'action': 0,
                'balance': self._capital,
                'total':total,
                'timestamp': str(datetime.now()),
            }

    def update_realtime_record_with_action(self, action, record):
      if action == "sell" and len(self._inventory):
        real_bought_price = self._inventory.pop(0)
        self._capital += record['Close']
        _scaled_capital = self.minmax.transform([[self._capital] * self.num_features])[0, 0]
        self._scaled_capital = _scaled_capital + self.minmax.transform([[record['Close']] * self.num_features])[0, 0]

      elif action == "buy":
        self._inventory.append(record['Close'])
        self._capital += record['Close']
        _scaled_capital = self.minmax.transform([[self._capital] * self.num_features])[0, 0]
        self._scaled_capital = _scaled_capital + self.minmax.transform([[record['Close']] * self.num_features])[0, 0]

    def change_data(self, timeseries, skip, initial_money, real_trend, minmax):
        self.timeseries = timeseries
        self.skip = skip
        self.initial_money = initial_money
        self.real_trend = real_trend
        self.minmax = minmax
        self._initiate()

    def act(self, sequence):
        decision = self.model.predict(np.array(sequence))

        return np.argmax(decision[0])

    def act_softmax(self, sequence):
        decision = self.model.predict(np.array(sequence))

        return np.argmax(decision[0]), softmax(decision)[0]

    def get_state(self, t, inventory, capital, timeseries):
        state = get_state(timeseries, t, window_size = self.window_size)
        len_inventory = len(inventory)
        if len_inventory:
            mean_inventory = np.mean(inventory)
        else:
            mean_inventory = 0
        z_inventory = (mean_inventory - self._mean) / self._std
        z_capital = (capital - self._mean) / self._std
        concat_parameters = np.concatenate(
            [state, [[len_inventory, z_inventory, z_capital]]], axis = 1
        )
        return concat_parameters

    def get_reward(self, weights):
        initial_money = self._scaled_capital
        starting_money = initial_money
        invests = []
        self.model.weights = weights
        inventory = []
        state = self.get_state(0, inventory, starting_money, self.timeseries)

        for t in range(0, len(self.trend) - 1, self.skip):
            action = self.act(state)
            if action == 1 and starting_money >= self.trend[t]:
                inventory.append(self.trend[t])
                starting_money -= self.trend[t]

            elif action == 2 and len(inventory):
                bought_price = inventory.pop(0)
                starting_money += self.trend[t]
                invest = ((self.trend[t] - bought_price) / bought_price) * 100
                invests.append(invest)

            state = self.get_state(
                t + 1, inventory, starting_money, self.timeseries
            )
        invests = np.mean(invests)
        if np.isnan(invests):
            invests = 0
        score = (starting_money - initial_money) / initial_money * 100
        return invests * 0.7 + score * 0.3

    def fit(self, iterations, checkpoint):
        self.es.train(iterations, print_every = checkpoint)

    def buy(self):
        initial_money = self._scaled_capital
        starting_money = initial_money

        real_initial_money = self.initial_money
        real_starting_money = self.initial_money
        inventory = []
        real_inventory = []
        state = self.get_state(0, inventory, starting_money, self.timeseries)
        states_sell = []
        states_buy = []

        for t in range(0, len(self.trend) - 1, self.skip):
            action, prob = self.act_softmax(state)
            print(t, prob)

            if action == 1 and starting_money >= self.trend[t] and t < (len(self.trend) - 1 - window_size):
                inventory.append(self.trend[t])
                real_inventory.append(self.real_trend[t])
                real_starting_money -= self.real_trend[t]
                starting_money -= self.trend[t]
                states_buy.append(t)
                print(
                    'day %d: buy 1 unit at price %f, total balance %f'
                    % (t, self.real_trend[t], real_starting_money)
                )

            elif action == 2 and len(inventory):
                bought_price = inventory.pop(0)
                real_bought_price = real_inventory.pop(0)
                starting_money += self.trend[t]
                real_starting_money += self.real_trend[t]
                states_sell.append(t)
                try:
                    invest = (
                        (self.real_trend[t] - real_bought_price)
                        / real_bought_price
                    ) * 100
                except:
                    invest = 0
                print(
                    'day %d, sell 1 unit at price %f, investment %f %%, total balance %f,'
                    % (t, self.real_trend[t], invest, real_starting_money)
                )
            state = self.get_state(
                t + 1, inventory, starting_money, self.timeseries
            )

        invest = (
            (real_starting_money - real_initial_money) / real_initial_money
        ) * 100
        total_gains = real_starting_money - real_initial_money
        return states_buy, states_sell, total_gains, invest


class LSTM_Model(nn.Module):
  def __init__(self, input_size = 1, output_size = 1, hidden_layer_size = 128, num_rnn_layers = 2, dropout = 0.2):
    super().__init__()
    self.hidden_layer_size = hidden_layer_size

    self.rnn = nn.LSTM(input_size, hidden_size = self.hidden_layer_size,
                       num_layers= num_rnn_layers, dropout = dropout, batch_first = True)
    self.fc = nn.Linear(num_rnn_layers* self.hidden_layer_size, output_size)

    self.init_weights()

  def init_weights(self):
    for name, m in self.named_modules():
      if isinstance(m, nn.Linear):
          torch.nn.init.xavier_uniform_(m.weight)
          m.bias.data.fill_(0.01)
      for name, param in self.rnn.named_parameters():
          if 'bias' in name:
                nn.init.constant_(param, 0.0)
          elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
          elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

  def forward(self, x):
    lstm_out, (h_n, le)  = self.rnn(x)

    #prediction = self.fc(torch.flatten(h_n.permute(1, 0, 2), start_dim = 1))
    prediction = self.fc(torch.flatten(h_n.permute(1, 0, 2), start_dim = 1))

    return prediction


class GRU_LSTM_Model(nn.Module):
    def __init__(self, input_size=1, output_size=1, hidden_layer_size=128, num_gru_layers=2, num_rnn_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        # GRU layers
        self.gru = nn.GRU(input_size, hidden_size=self.hidden_layer_size,
                          num_layers=num_gru_layers, dropout=dropout, batch_first=True)
        
        # LSTM layers
        self.rnn = nn.LSTM(self.hidden_layer_size, hidden_size=self.hidden_layer_size,
                           num_layers=num_rnn_layers, dropout=dropout, batch_first=True)
        
        self.fc = nn.Linear(num_rnn_layers * self.hidden_layer_size, output_size)

        self.init_weights()

    def init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x):
        # Apply GRU
        gru_out, h_n = self.gru(x)

        # Apply LSTM
        lstm_out, (h_n, le) = self.rnn(gru_out)

        # Apply fully connected layer
        prediction = self.fc(torch.flatten(h_n.permute(1, 0, 2), start_dim=1))

        return prediction



class Attention(nn.Module):
    def __init__(self, hidden_layer_size):
        super(Attention, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.attn = nn.Linear(hidden_layer_size * 2, hidden_layer_size)
        self.v = nn.Parameter(torch.rand(hidden_layer_size))

    def forward(self, hidden, encoder_outputs):
        # hidden: (num_layers * num_directions, batch_size, hidden_layer_size)
        # encoder_outputs: (batch_size, seq_len, hidden_layer_size)
        timestep = encoder_outputs.size(1)
        h = hidden[-1].unsqueeze(1).repeat(1, timestep, 1)  # (batch_size, seq_len, hidden_layer_size)
        attn_energies = self.score(h, encoder_outputs)  # (batch_size, seq_len)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)  # (batch_size, 1, seq_len)

    def score(self, hidden, encoder_outputs):
        # hidden: (batch_size, seq_len, hidden_layer_size)
        # encoder_outputs: (batch_size, seq_len, hidden_layer_size)
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))  # (batch_size, seq_len, hidden_layer_size)
        energy = energy.transpose(1, 2)  # (batch_size, hidden_layer_size, seq_len)
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # (batch_size, 1, hidden_layer_size)
        energy = torch.bmm(v, energy)  # (batch_size, 1, seq_len)
        return energy.squeeze(1)  # (batch_size, seq_len)
    
class LSTM_Attention_Model(nn.Module):
    def __init__(self, input_size=1, output_size=1, hidden_layer_size=128, num_rnn_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.rnn = nn.LSTM(input_size, hidden_size=self.hidden_layer_size,
                           num_layers=num_rnn_layers, dropout=dropout, batch_first=True)
        self.attention = Attention(hidden_layer_size)
        self.fc = nn.Linear(hidden_layer_size, output_size)

        self.init_weights()

    def init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            for name, param in self.rnn.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight_ih' in name:
                    nn.init.kaiming_normal_(param)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.rnn(x)

        attn_weights = self.attention(h_n, lstm_out)  # (batch_size, 1, seq_len)
        context = torch.bmm(attn_weights, lstm_out)  # (batch_size, 1, hidden_layer_size)
        context = context.squeeze(1)  # (batch_size, hidden_layer_size)

        prediction = self.fc(context)  # (batch_size, output_size)

        return prediction



class VAE(nn.Module):
    def __init__(self, config, latent_dim):
        super().__init__()

        modules = []
        for i in range(1, len(config)):
            modules.append(
                nn.Sequential(
                    nn.Linear(config[i - 1], config[i]),
                    nn.ReLU()
                )
            )

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(config[-1], latent_dim)
        self.fc_var = nn.Linear(config[-1], latent_dim)

        modules = []
        self.decoder_input = nn.Linear(latent_dim, config[-1])

        for i in range(len(config) - 1, 1, -1):
            modules.append(
                nn.Sequential(
                    nn.Linear(config[i], config[i - 1]),
                    nn.ReLU()
                )
            )
        modules.append(
            nn.Sequential(
                nn.Linear(config[1], config[0]),
                nn.Sigmoid()
            )
        )

        self.decoder = nn.Sequential(*modules)

    def encode(self, x):
        result = self.encoder(x)
        mu = self.fc_mu(result)
        logVar = self.fc_var(result)
        return mu, logVar

    def decode(self, x):
        result = self.decoder(x)
        return result

    def reparameterize(self, mu, logVar):
        std = torch.exp(0.5* logVar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        mu, logVar = self.encode(x)
        z = self.reparameterize(mu, logVar)
        output = self.decode(z)
        return output, z, mu, logVar

class Generator(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.gru_1 = nn.GRU(input_size, 1024, batch_first = True)
        self.gru_2 = nn.GRU(1024, 512, batch_first = True)
        self.gru_3 = nn.GRU(512, 256, batch_first = True)
        self.linear_1 = nn.Linear(256, 128)
        self.linear_2 = nn.Linear(128, 64)
        self.linear_3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)


    def forward(self, x):
        use_cuda = 1
        device = torch.device("cuda" if (torch.cuda.is_available() & use_cuda) else "cpu")
        h0 = torch.zeros(1, x.size(0), 1024).to(device)
        out_1, _ = self.gru_1(x, h0)
        out_1 = self.dropout(out_1)
        h1 = torch.zeros(1, x.size(0), 512).to(device)
        out_2, _ = self.gru_2(out_1, h1)
        out_2 = self.dropout(out_2)
        h2 = torch.zeros(1, x.size(0), 256).to(device)
        out_3, _ = self.gru_3(out_2, h2)
        out_3 = self.dropout(out_3)
        out_4 = self.linear_1(out_3[:, -1, :])
        out_5 = self.linear_2(out_4)
        out_6 = self.linear_3(out_5)
        return out_6

class Discriminator(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size = 5, stride = 1, padding = 'same')
        self.conv2 = nn.Conv1d(32, 64, kernel_size = 5, stride = 1, padding = 'same')
        self.conv3 = nn.Conv1d(64, 128, kernel_size = 5, stride = 1, padding = 'same')
        self.linear1 = nn.Linear(128, 220)
        self.linear2 = nn.Linear(220, 220)
        self.linear3 = nn.Linear(220, 1)
        self.leaky = nn.LeakyReLU(0.01)
        self.relu = nn.ReLU()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv1 = self.leaky(conv1)
        conv2 = self.conv2(conv1)
        conv2 = self.leaky(conv2)
        conv3 = self.conv3(conv2)
        conv3 = self.leaky(conv3)
        flatten_x =  conv3.reshape(conv3.shape[0], conv3.shape[1])
        out_1 = self.linear1(flatten_x)
        out_1 = self.leaky(out_1)
        out_2 = self.linear2(out_1)
        out_2 = self.relu(out_2)
        out = self.linear3(out_2)
        return out
