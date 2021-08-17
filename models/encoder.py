from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, dropout_encoder):
        super().__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=n_layers, dropout=dropout_encoder)

    def forward(self, inpput_):
        outputs, (hidden, cell) = self.rnn(inpput_)
        return hidden, cell
