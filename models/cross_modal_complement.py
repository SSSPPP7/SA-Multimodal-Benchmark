import torch, torch.nn as nn

class CrossModalComplement(nn.Module):
    def __init__(self,dim=64,num_heads=1,dropout=0.2):
        super().__init__()
        self.eeg_from_em=nn.MultiheadAttention(dim,num_heads,dropout=dropout,batch_first=True)
        self.em_from_eeg=nn.MultiheadAttention(dim,num_heads,dropout=dropout,batch_first=True)
        self.ne=nn.LayerNorm(dim); self.nm=nn.LayerNorm(dim); self.drop=nn.Dropout(dropout)
    def forward(self,f_eeg,f_em):
        e=f_eeg.transpose(1,2); m=f_em.transpose(1,2)
        ec,ae=self.eeg_from_em(e,m,m,need_weights=True,average_attn_weights=True)
        mc,am=self.em_from_eeg(m,e,e,need_weights=True,average_attn_weights=True)
        e2=self.ne(e+self.drop(ec)); m2=self.nm(m+self.drop(mc))
        return e2.transpose(1,2),m2.transpose(1,2),{'eeg_from_em':ae,'em_from_eeg':am}
