"""
Github: xichuanhit
https://github.com/xichuanhit/PerCode/tree/main/Stock-Pre

models include 'lstm', 'fnn' model

environment: python=3.9.7, torch=1.10.1
"""

import numpy as np
import torch
from torch import nn
import tushare as ts
import matplotlib.pyplot as plt

import models

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

DAYS_TRAIN = 5              # the days for training stock
STOCK_CODE = '600519'       # stock code
TEST_LEN = 300              # the len of test set
SATRT_DATE = '2018-01-01'   # start date of stock

class Solver(object):
    def __init__(self, name):
        self.batch_size = 64
        self.learning_rate = 0.01
        self.epochs = 500

        self.name = name
        
    def model(self, name):
        if name == 'lstm':
            model = models.LSTM(input_size=1, hidden_layer_size=100, days_train=DAYS_TRAIN, output_size=1)
        elif name == 'fnn':
            model = models.FNN(input_size=DAYS_TRAIN, output_size=1)
        return model

    def Data(self, code):
        print("\n---------Begin loading data..---------\n")
        stock = ts.get_k_data(code, ktype='D', start=SATRT_DATE)['open'].values
        print("\n---------Data loading is done---------\n")
        stock = stock.astype('float32')

        self.max = np.max(stock)
        self.min = np.min(stock)

        stock = (stock - self.min) / (self.max - self.min)

        train_data, label_data = [], []
        for i in range(DAYS_TRAIN, stock.shape[0]):
            train_data.append(stock[i-DAYS_TRAIN:i])
            label_data.append(stock[i])

        train_data = np.array(train_data)
        label_data = np.array(label_data)

        if self.name == 'lstm':
            train_data = np.reshape(train_data, (train_data.shape[0], 1, DAYS_TRAIN))
        elif self.name == 'fnn':
            rain_data = np.reshape(train_data, (train_data.shape[0], DAYS_TRAIN))

        label_data = np.reshape(label_data, (label_data.shape[0], 1))

        return torch.Tensor(train_data), torch.Tensor(label_data)

    def train(self,):

        train_data, label_data = self.Data(STOCK_CODE)
        x_train, y_train = train_data[:-TEST_LEN], label_data[:-TEST_LEN]
        model = self.model(self.name)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criteon = nn.MSELoss()
        
        print('------------Begin training------------\n')
        for epoch in range(self.epochs):
            out = model(x_train)
            loss = criteon(out, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (epoch+1) % 100 == 0:
                print('Train Epoch : {} \tLoss: {:.6f}'.format(epoch+1, loss.item()))

        model_test = model.eval()
        x_test, y_test = train_data[-TEST_LEN:], label_data[-TEST_LEN:]
        pred_test = model_test(x_test)
        pred_data = model_test(train_data)
        test_loss = criteon(pred_test, y_test)
        print("Test loss : {:.6f}".format(test_loss))

        pred_data = pred_data.view(-1).detach().numpy()
        label_data = label_data.view(-1).detach().numpy()

        size = len(y_train)

        pred_data = pred_data*(self.max - self.min) + self.min
        label_data = label_data*(self.max - self.min) + self.min

        plt.plot(pred_data, 'r', label='Prediction')
        plt.plot(label_data, 'b', label='Real')
        plt.vlines(size, self.min, self.max, colors="g", linestyles="dashed")
        plt.legend(loc='best')
        plt.savefig(self.name + '_result.pdf', dpi=1000)
        plt.close()

        print('\n---------Forcasting is done !---------')


def main():
    # models include 'lstm', 'fnn'
    solver = Solver('fnn')
    solver.train()


if __name__ == '__main__':
    main()

