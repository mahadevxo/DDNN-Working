import torch.nn as nn
import torch

class LearnData(nn.Module):
    def __init__(self):
        super(LearnData, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
    
    def forward(self, x):
        x = self.net(x)
        return torch.abs(x)