import torch
from torch import nn


class VelDecoder(nn.Module):
    def __init__(self, outputs_num, input_size, output_size, hardtanh_limit, hidden_size, n_layers, dropout_pose_dec):
        super().__init__()
        self.outputs_num = outputs_num
        self.dropout = nn.Dropout(dropout_pose_dec)
        rnns = [
            nn.LSTMCell(input_size=input_size if i == 0 else hidden_size, hidden_size=hidden_size).cuda() for
            i in range(n_layers)]
        self.rnns = nn.Sequential(*rnns)
        self.fc_out = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.hardtanh = nn.Hardtanh(min_val=-1 * hardtanh_limit, max_val=hardtanh_limit, inplace=False)

    def forward(self, inputs, hiddens, cells):
        dec_inputs = self.dropout(inputs)
        if len(hiddens.shape) < 3 or len(cells.shape) < 3:
            hiddens = torch.unsqueeze(hiddens, 0)
            cells = torch.unsqueeze(cells, 0)
        outputs = torch.tensor([], device='cuda')
        for j in range(self.outputs_num):
            for i, rnn in enumerate(self.rnns):
                if i == 0:
                    hiddens[i], cells[i] = rnn(dec_inputs, (hiddens.clone()[i], cells.clone()[i]))
                else:
                    hiddens[i], cells[i] = rnn(hiddens.clone()[i - 1], (hiddens.clone()[i], cells.clone()[i]))
            output = self.hardtanh(self.fc_out(hiddens.clone()[-1]))
            dec_inputs = output.detach()
            outputs = torch.cat((outputs, output.unsqueeze(1)), dim=1)
        return outputs


class MaskDecoder(nn.Module):
    def __init__(self, args, input_size, output_size):
        super().__init__()
        self.args = args
        self.dropout = nn.Dropout(self.args.dropout_mask_dec)
        rnns = [
            nn.LSTMCell(input_size=input_size if i == 0 else args.hidden_size, hidden_size=args.hidden_size).cuda() for
            i in range(args.n_layers)]
        self.rnns = nn.Sequential(*rnns)
        self.fc_out = nn.Linear(in_features=args.hidden_size, out_features=output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, hiddens, cells):
        dec_input = self.dropout(inputs)
        if len(hiddens.shape) < 3 or len(cells.shape) < 3:
            hiddens = torch.unsqueeze(hiddens, 0)
            cells = torch.unsqueeze(cells, 0)
        outputs = torch.tensor([], device=self.args.device)
        for j in range(self.args.output):
            for i, rnn in enumerate(self.rnns):
                if i == 0:
                    hiddens[i], cells[i] = rnn(dec_input, (hiddens.clone()[i], cells.clone()[i]))
                else:
                    hiddens[i], cells[i] = rnn(hiddens.clone()[i - 1], (hiddens.clone()[i], cells.clone()[i]))
            output = self.sigmoid(self.fc_out(hiddens.clone()[-1]))
            dec_input = output.detach()
            outputs = torch.cat((outputs, output.unsqueeze(1)), dim=1)
        return outputs
