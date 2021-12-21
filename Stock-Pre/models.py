"""
Github: xichuanhit
https://github.com/xichuanhit/PerCode/tree/main/Stock-Pre

models include 'lstm', 'fnn' model

environment: python=3.9.7, torch=1.10.1
"""

import numpy as np
import torch
from torch import nn
from torch import functional as F

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class LSTM(torch.nn.Module):
    
    def __init__(self, input_size=1, hidden_layer_size=100, days_train=10, output_size=1):
        super(LSTM,self).__init__()
        self.days_train = days_train
        self.lstm = nn.LSTM(input_size, hidden_layer_size, 2)
        self.ln1 = nn.Linear(hidden_layer_size*days_train, hidden_layer_size)
        self.ln2 = nn.Linear(hidden_layer_size, output_size)
                            
    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq.view(len(input_seq), self.days_train, -1))
        out = self.ln1(lstm_out.view(len(input_seq),-1))
        out = self.ln2(out)
        return out


class FNN(torch.nn.Module):

    def __init__(self, input_size, hidden_layer_size=100, output_size=1):
        super(FNN,self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, output_size)
        )
                            
    def forward(self, input_seq):
        out = self.fc(input_seq)
        return out

