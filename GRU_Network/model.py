import torch
from torch import nn

from torch import nn

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, num_gru=5):
        super(GRUModel, self).__init__()
        self.num_gru = num_gru
        self.gru_layers = nn.ModuleList([nn.GRU(input_size, hidden_size, num_layers, batch_first=True) for i in range(num_gru-1)])
        self.fc = nn.Linear(hidden_size, output_size)
        self.criterion = nn.MSELoss()

    def forward(self, x, z):
        batch_size, seq_length, input_size = x.size()
        total_loss = 0
        out = x[:,0:2,:]

        for i, gru in enumerate(self.gru_layers):
            if i == 0:
                target = x[:, 2, :]
                out, _ = gru(out)  # First GRU takes first two inputs
                out = out[:, -1, :]
                out = self.fc(out).unsqueeze(1)
                total_loss = total_loss + self.criterion(out[:, -1, :], target)
            elif i<self.num_gru-2:
                target = x[:, i+2, :]
                y = x[:,i+1:i+2,:]
                out = torch.cat((y, out), dim=1)
                out, _ = gru(out)  # Subsequent GRUs take output of previous GRU
                out = out[:, -1, :]
                out = self.fc(out).unsqueeze(1)
                total_loss = total_loss + self.criterion(out[:, -1, :], target)

        y = x[:, i + 1:i + 2, :]
        out = torch.cat((y, out), dim=1)
        out, _ = gru(out)  # Subsequent GRUs take output of previous GRU
        out = out[:, -1, :]
        out = self.fc(out).unsqueeze(1)
        total_loss = total_loss + self.criterion(out[:, -1, :], z)
        out = out[:, -1, :]

        return out, total_loss

    def get_output(self, x):
        out = x[:, 0:2, :]

        for i, gru in enumerate(self.gru_layers):
            if i == 0:
                target = x[:, 2, :]
                out, _ = gru(out)  # First GRU takes first two inputs
                out = out[:, -1, :]
                out = self.fc(out).unsqueeze(1)
                # total_loss = total_loss + self.criterion(out[:, -1, :], target)
            elif i < self.num_gru - 2:
                target = x[:, i + 2, :]
                y = x[:, i + 1:i + 2, :]
                out = torch.cat((y, out), dim=1)
                out, _ = gru(out)  # Subsequent GRUs take output of previous GRU
                out = out[:, -1, :]
                out = self.fc(out).unsqueeze(1)
                # total_loss = total_loss + self.criterion(out[:, -1, :], target)

        y = x[:, i + 1:i + 2, :]
        out = torch.cat((y, out), dim=1)
        out, _ = gru(out)  # Subsequent GRUs take output of previous GRU
        out = out[:, -1, :]
        out = self.fc(out).unsqueeze(1)
        out = out[:, -1, :]
        return out