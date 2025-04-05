import numpy as np
from torch import nn

class gyro_model_RNN(nn.Module):
    def __init__(self, input_size, sequence_length, hidden_size, num_layers, output_size, dropout_prob ):
        super(gyro_model_RNN, self).__init__()

        self.input_size = input_size
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True,
                            dropout=dropout_prob if self.num_layers > 1 else 0)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, self.output_size) 
        )

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden_state = h_n[-1]
        x = self.fc_layers(last_hidden_state)
        return x