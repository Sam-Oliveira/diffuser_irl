import torch.nn as nn
import torch.nn.functional as F


class Guide_Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Guide_Net, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        
        
    def forward(self, x):
        x = F.relu(self.i2h(x))
        x = F.relu(self.h2h(x))
        x = F.relu(self.h2o(x))
        return x