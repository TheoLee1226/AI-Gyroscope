import numpy as np
from torch import nn

class gyro_model(nn.Module):
    def __init__(self, input_size, sequence_length, output_size):
        super(gyro_model, self).__init__()
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.output_size = output_size
        
        self.conv_layers = nn.Sequential(
            # (N, 6, 1000) -> (N, 32, 1000)
            nn.Conv1d(in_channels=self.input_size, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=32),
            # (N, 32, 1000) -> (N, 32, 500)
            nn.MaxPool1d(kernel_size=2, stride=2),
            # (N, 32, 500) -> (N, 64, 500)
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=64),
            # (N, 64, 500) -> (N, 64, 250)
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        flattened_size = 64 * (self.sequence_length // 4)

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            # (N, 64, 250) -> (N, 64 * 250)
            nn.Linear(flattened_size, 128), 
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(128, self.output_size) 
        )

    def forward(self, x):
        # (N, 1000, 6) -> (N, 6, 1000)
        x = x.permute(0, 2, 1)
        x = self.conv_layers(x)
        x = self.fc_layers(x) 
        return x
