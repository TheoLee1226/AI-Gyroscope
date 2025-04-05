import numpy as np
from torch import nn

class gyro_model(nn.Model):
    def __init__(self, input_size, output_size):
        super(gyro_model, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
        
    def forward(self, x):
        return self.layers(x)
