import torch
import torch.nn as nn


class LSTM1D(nn.Module):
    def __init__(self, tdim):
        super(LSTM1D, self).__init__()
        self.gru1 = nn.GRU(input_size=1, hidden_size=64, batch_first=True)
        self.out = nn.Linear(64 * tdim, 2)
        self.relu = nn.ReLU()
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.kaiming_normal_(self.out.weight)        
    
    def forward(self, x):
        x = x.float()
        x = x.permute(0, 2, 1)  # Swap the second and third dimensions
        x, _ = self.gru1(x)
        x = torch.flatten(x, 1)
        return self.out(x)


class LSTM2D(nn.Module):
    def __init__(self, fdim, tdim):
        super(LSTM2D, self).__init__()
        self.gru1 = nn.GRU(input_size=fdim, hidden_size=64, batch_first=True)
        self.out = nn.Linear(64 * tdim, 2)
        self.relu = nn.ReLU()
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.kaiming_normal_(self.out.weight)        
    
    def forward(self, x):
        x = x.squeeze(1).float()
        x = x.permute(0, 2, 1)  # Swap the second and third dimensions
        x, _ = self.gru1(x)
        x = torch.flatten(x, 1)
        return self.out(x)