import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class HAS_BS(nn.Module):
    def __init__(self, in_channels=15):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)

        self.deconv4 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv1 = nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1)

        self.attn1 = nn.Linear(256, in_channels)
        self.attn2 = nn.Linear(128, in_channels)
        self.attn3 = nn.Linear(64, in_channels)
        self.attn4 = nn.Linear(32, in_channels)
        self.attn5 = nn.Linear(in_channels, in_channels)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.initialize_parameters()

    def initialize_parameters(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0.0)

    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        f4 = self.conv4(f3)
        attention_scores1 = self.attn1(f4.mean(dim=[2, 3]))
        attention_weights1 = F.softmax(attention_scores1, dim=1)

        d4 = self.deconv4(f4)
        d4 = d4 + self.dropout1(f3)
        attention_scores2 = self.attn2(d4.mean(dim=[2, 3]))
        attention_weights2 = F.softmax(attention_scores2, dim=1)
        d3 = self.deconv3(d4)
        d3 = d3 + self.dropout2(f2)
        attention_scores3 = self.attn3(d3.mean(dim=[2, 3]))
        attention_weights3 = F.softmax(attention_scores3, dim=1)
        d2 = self.deconv2(d3)
        d2 = d2 + self.dropout3(f1)
        attention_scores4 = self.attn4(d2.mean(dim=[2, 3]))
        attention_weights4 = F.softmax(attention_scores4, dim=1)
        d1 = self.deconv1(d2)
        # d1 = d1 + self.dropout4(x)
        attention_scores5 = self.attn5(d1.mean(dim=[2, 3]))
        attention_weights5 = F.softmax(attention_scores5, dim=1)
        attention_weight = attention_weights1 + attention_weights2 + attention_weights3 + attention_weights4 + attention_weights5
        sorted_indices = torch.argsort(attention_weight, descending=True)
        return d1, sorted_indices

if __name__ == "__main__":
    input = torch.randn((1, 15, 224, 224))
    babs = HAS_BS(in_channels=15)
    re_image, order = babs(input)
    print(order)