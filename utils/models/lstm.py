import torch
import torch.nn as nn

'''Code from https://www.kaggle.com/code/fanbyprinciple/learning-pytorch-3-coding-an-rnn-gru-lstm'''
class LSTM(nn.Module):
    input_size=16
    window_length=10
    num_layers=2
    hidden_size=256
    def __init__(self, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, window_length=window_length):
        super(LSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size * window_length, 1)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x,(h0, c0))
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        return out