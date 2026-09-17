import torch
from torch import nn

class NormalizationLayer(nn.Module):
    def __init__(self, x_interval, t_interval):
        super(NormalizationLayer, self).__init__()
        self.register_buffer("x_min", torch.tensor(x_interval[0]))
        self.register_buffer("x_max", torch.tensor(x_interval[1]))
        self.register_buffer("t_min", torch.tensor(t_interval[0]))
        self.register_buffer("t_max", torch.tensor(t_interval[1]))

    def forward(self, inputs):
        t = inputs[:, 0:1]
        x = inputs[:, 1:2]
        x_normalized = 2 * (x - self.x_min) / (self.x_max - self.x_min) - 1
        t_normalized = 2 * (t - self.t_min) / (self.t_max - self.t_min) - 1
        omega = 2 * torch.pi
        return torch.cat((t_normalized, x_normalized*omega), dim=1)
    
class ClassicalDNN(nn.Module):
    def __init__(self, x_interval=(0.0, 4.0 * torch.pi), t_interval=(0.0, 3.0)):
        super(ClassicalDNN, self).__init__()
        self.model = nn.Sequential(
            NormalizationLayer(x_interval=x_interval, t_interval=t_interval),
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)
