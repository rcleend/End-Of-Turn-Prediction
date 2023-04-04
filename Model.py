import torch
class LSTMNet(torch.nn.Module):
    def __init__(self, n_input, n_hidden):
        self.lstm = torch.nn.LSTM(n_input, n_hidden)

    def forward(self, x):
        # Input features to LSTM
        out_lstm, _ = self.lstm(x)
        # Get probabilities for N frames if speaker is talking or not using sigmoid activation
        return out_lstm
