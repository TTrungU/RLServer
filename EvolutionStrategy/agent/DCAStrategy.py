import numpy as np
import time
import math
from datetime import timedelta
import torch
import torch.nn as nn
from datetime import datetime
from utils import softmax, get_state
from modules import Deep_Evolution_Strategy, Model



class DCAAgent:

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

    def reset_capital(self, capital):
        if capital:
            self._capital = capital
        self._scaled_capital = self.minmax.transform([[self._capital] * self.num_features])[0, 0]
        self._queue = []
        self._inventory = []

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
                'status': 'data not enough to trade',
                'action': 'fail',
                'balance': self._capital,
                'timestamp': str(datetime.now()),
                'date': date,
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
            return {
                'status': 'buy 1 unit, cost %f' % (real_close),
                'action': 'buy',
                'balance': self._capital,
                'timestamp': str(datetime.now()),
                'date': date,
            }
        elif action == 2 and len(self._inventory):
            total_investment_return = 0
            total_gain = 0
            total_units = len(self._inventory)
            while len(self._inventory) > 0:
                bought_price = self._inventory.pop(0)
                scaled_bought_price = self.minmax.inverse_transform(
                    [[bought_price] * self.num_features]
                )[0, 0]
                try:
                    invest = (
                        (real_close - scaled_bought_price) / scaled_bought_price
                    ) * 100
                except:
                    invest = 0

                total_investment_return += invest
                total_gain += real_close - scaled_bought_price


            self._scaled_capital += close * total_units
            self._capital += real_close *  total_units
             # Calculate average investment return
            average_investment_return = total_investment_return / total_units if total_units > 0 else 0
            return {
                'status': 'sold %d units, price %f' % (total_units, real_close),
                'total_investment': total_investment_return,
                'average_investment': average_investment_return,
                'total_gain': total_gain,
                'balance': self._capital,
                'action': 'sell',
                'timestamp': str(datetime.now()),
                'date': date,
            }
        else:
            return {
                'status': 'do nothing',
                'action': 'nothing',
                'balance': self._capital,
                'timestamp': str(datetime.now()),
            }

    def update_realtime_record_with_action(self, action, record):
      if action == "sell" and len(self._inventory):
        # real_bought_price = self._inventory.pop(0)
        # self._capital += record['Close']
        # _scaled_capital = self.minmax.transform([[self._capital] * self.num_features])[0, 0]
        # self._scaled_capital = _scaled_capital + self.minmax.transform([[record['Close']] * self.num_features])[0, 0]
        total_units = len(self._inventory)
        total_gain = 0
        total_real_bought_price = 0

        while len(self._inventory) > 0:
            real_bought_price = self._inventory.pop(0)
            total_real_bought_price += real_bought_price
            total_gain += record['Close'] - real_bought_price

        self._capital += record['Close'] * total_units
        
        # Calculate scaled capital for each unit and update scaled capital
        scaled_record_close = self.minmax.transform([[record['Close']] * self.num_features])[0, 0]
        _scaled_capital = self.minmax.transform([[self._capital] * self.num_features])[0, 0]
        self._scaled_capital = _scaled_capital + scaled_record_close * total_units

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
                # Use 50% of the available capital to buy
                amount_to_use = 0.5 * starting_money
                units_to_buy = int(amount_to_use // self.trend[t])
            
                if units_to_buy > 0:
                    for _ in range(units_to_buy):
                        inventory.append(self.trend[t])
                        starting_money -= self.trend[t]

            elif action == 2 and len(inventory):
                # bought_price = inventory.pop(0)
                # starting_money += self.trend[t]
                # invest = ((self.trend[t] - bought_price) / bought_price) * 100
                # invests.append(invest)
                total_units = len(inventory)
                total_gain = 0
                total_investment_return = 0
                while len(inventory) > 0:
                    bought_price = inventory.pop(0)
                    total_gain += self.trend[t] - bought_price
                    total_investment_return += ((self.trend[t] - bought_price) / bought_price) * 100
                
                average_investment_return = total_investment_return / total_units if total_units > 0 else 0
                starting_money += self.trend[t] * total_units
                invests.append(average_investment_return)


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
                # bought_price = inventory.pop(0)
                # real_bought_price = real_inventory.pop(0)
                # starting_money += self.trend[t]
                # real_starting_money += self.real_trend[t]
                # states_sell.append(t)
                # try:
                #     invest = (
                #         (self.real_trend[t] - real_bought_price)
                #         / real_bought_price
                #     ) * 100
                # except:
                #     invest = 0
                # print(
                #     'day %d, sell 1 unit at price %f, investment %f %%, total balance %f,'
                #     % (t, self.real_trend[t], invest, real_starting_money)
                # )
                    # Sell all units at once
                total_units = len(inventory)
                total_investment_return = 0
                total_gain = 0

                while len(inventory) > 0:
                    bought_price = inventory.pop(0)
                    real_bought_price = real_inventory.pop(0)
                    total_investment_return += ((self.real_trend[t] - real_bought_price) / real_bought_price) * 100
                    total_gain += self.real_trend[t] - real_bought_price

                average_investment_return = total_investment_return / total_units if total_units > 0 else 0
                starting_money += self.trend[t] * total_units
                real_starting_money += self.real_trend[t] * total_units
                states_sell.append(t)

                print(
                    'day %d, sell %d units at price %f, average investment %f %%, total balance %f'
                    % (t, total_units, self.real_trend[t], average_investment_return, real_starting_money)
                )
            state = self.get_state(
                t + 1, inventory, starting_money, self.timeseries
            )

        invest = (
            (real_starting_money - real_initial_money) / real_initial_money
        ) * 100
        total_gains = real_starting_money - real_initial_money
        return states_buy, states_sell, total_gains, invest
