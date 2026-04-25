import torch, torch.nn as nn

def pad(k): return k//2

class ConvBNELUDrop(nn.Module):
    def __init__(self,cin,cout,k,drop):
        super().__init__(); self.net=nn.Sequential(nn.Conv1d(cin,cout,k,padding=pad(k),bias=False),nn.BatchNorm1d(cout),nn.ELU(inplace=True),nn.Dropout(drop))
    def forward(self,x): return self.net(x)

class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self,c,k=3,dropout=0.2):
        super().__init__(); self.net=nn.Sequential(nn.Conv1d(c,c,k,padding=pad(k),groups=c,bias=False),nn.Conv1d(c,c,1,bias=False),nn.BatchNorm1d(c),nn.ELU(inplace=True),nn.Dropout(dropout))
    def forward(self,x): return self.net(x)

class EEGEncoder(nn.Module):
    def __init__(self,in_channels=32,hidden_dim=64,branch_channels=16,kernels_block1=(64,32,16,8),kernels_block3=(16,8,4,2),dropout=0.2):
        super().__init__()
        self.ms1=nn.ModuleList([ConvBNELUDrop(in_channels,branch_channels,k,dropout) for k in kernels_block1])
        self.ds=nn.ModuleList([DepthwiseSeparableConv1d(branch_channels,3,dropout) for _ in kernels_block1])
        self.pool1=nn.AvgPool1d(2,2); cat=branch_channels*4
        self.ms3=nn.ModuleList([ConvBNELUDrop(cat,branch_channels,k,dropout) for k in kernels_block3])
        self.pool3=nn.AvgPool1d(2,2)
        self.proj=nn.Sequential(nn.Conv1d(cat,hidden_dim,1,bias=False),nn.BatchNorm1d(hidden_dim),nn.ELU(inplace=True),nn.Dropout(dropout))
        self.gap=nn.AdaptiveAvgPool1d(1)
    def forward(self,eeg):
        x=torch.cat([ds(conv(eeg)) for conv,ds in zip(self.ms1,self.ds)],1)
        x=self.pool1(x)
        x=torch.cat([conv(x) for conv in self.ms3],1)
        x=self.pool3(x); fmap=self.proj(x)
        return fmap, self.gap(fmap).squeeze(-1)
