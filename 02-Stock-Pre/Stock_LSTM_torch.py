"""
Github: xichuanhit
https://github.com/xichuanhit/PerCode/blob/main/02-Stock-Pre/Stock_LSTM_torch.py

environment: python=3.9.7, torch=1.10.1
"""

import numpy as np
import torch
from torch import nn
import tushare as ts
import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

DAYS_TRAIN = 5            # the days for training stock
STOCK_CODE = '600519'     # stock code
TEST_LEN = 300            # the len of test set

class Solver(object):
    def __init__(self, ):
        self.batch_size = 64
        self.learning_rate = 0.01
        self.epochs = 500

        self.model = Net()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criteon = nn.MSELoss()

    def Data(self, name):
        print("---------Begin loading data..---------")
        stock = ts.get_k_data(name, ktype='D', start='2018-01-01')['open'].values
        print("---------Data loading is done---------")
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

        train_data = np.reshape(train_data, (train_data.shape[0], 1, DAYS_TRAIN))
        label_data = np.reshape(label_data, (label_data.shape[0], 1))

        return torch.Tensor(train_data), torch.Tensor(label_data)

    def train(self, ):

        train_data, label_data = self.Data(STOCK_CODE)
        x_train, y_train = train_data[:-TEST_LEN], label_data[:-TEST_LEN]
        
        print('------------Begin training------------')
        for epoch in range(self.epochs):
            out = self.model(x_train)
            loss = self.criteon(out, y_train)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if (epoch+1) % 100 == 0:
                print('Train Epoch : {} \tLoss: {:.6f}'.format(epoch+1, loss.item()))

        model = self.model.eval()
        x_test, y_test = train_data[-TEST_LEN:], label_data[-TEST_LEN:]
        pred_test = model(x_test)
        pred_data = model(train_data)
        test_loss = self.criteon(pred_test, y_test)
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
        plt.savefig('lstm_result.pdf', dpi=1000)
        plt.close()

        print('---------Forcasting is done !---------')


class Net(torch.nn.Module):
    def __init__(self, input_size=1, hidden_layer_size= 100, output_size=1):
        super(Net,self).__init__()
        self.hidden_layer = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, 2)
        self.ln1 = nn.Linear(hidden_layer_size*DAYS_TRAIN, hidden_layer_size)
        self.ln2 = nn.Linear(hidden_layer_size, output_size)
                            
    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq.view(len(input_seq), DAYS_TRAIN, -1))
        pre = self.ln1(lstm_out.view(len(input_seq),-1))
        pre = self.ln2(pre)
        return pre


def main():
    solver = Solver()
    solver.train()


if __name__ == '__main__':
    main()
