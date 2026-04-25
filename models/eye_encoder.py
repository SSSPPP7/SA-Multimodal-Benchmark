import torch.nn as nn

class EyeEncoder(nn.Module):
    def __init__(self,in_channels=6,hidden_dim=64,conv_channels=64,dropout=0.2):
        super().__init__()
        self.net=nn.Sequential(nn.Conv1d(in_channels,conv_channels,7,padding=3,bias=False),nn.LeakyReLU(0.1,inplace=True),nn.Dropout(dropout),nn.AvgPool1d(2,2),
                               nn.Conv1d(conv_channels,hidden_dim,5,padding=2,bias=False),nn.LeakyReLU(0.1,inplace=True),nn.Dropout(dropout),nn.AvgPool1d(2,2))
        self.gap=nn.AdaptiveAvgPool1d(1)
    def forward(self,em):
        x=self.net(em)
        return x,self.gap(x).squeeze(-1)
