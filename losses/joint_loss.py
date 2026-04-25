import torch, torch.nn as nn
LEVELS=('sa1','sa2','sa3')

def _masked_bce(logits,target,mask=None,eps=1e-8):
    loss=nn.functional.binary_cross_entropy_with_logits(logits,target.float(),reduction='none')
    return loss.mean() if mask is None else (loss*mask.float()).sum()/(mask.float().sum()+eps)

class JointMDFSACNLoss(nn.Module):
    def __init__(self,lambda_main=1.0,lambda_aux=0.3,lambda_cg=0.1,contribution_detach=True,eps=1e-8):
        super().__init__(); self.lambda_main=lambda_main; self.lambda_aux=lambda_aux; self.lambda_cg=lambda_cg; self.det=contribution_detach; self.eps=eps
    def forward(self,outputs,batch):
        y=batch['y_rec'].float(); yp=batch['y_pred'].float(); mp=batch['mask_pred'].float(); mr=batch.get('mask_rec',torch.ones_like(y)).float()
        lrec=y.new_tensor(0.0); lpred=y.new_tensor(0.0); laux=y.new_tensor(0.0); lcg=y.new_tensor(0.0)
        for i,l in enumerate(LEVELS):
            lrec=lrec+_masked_bce(outputs['rec_logits'][l],y[:,i],mr[:,i],self.eps)
            lpred=lpred+_masked_bce(outputs['pred_logits'][l],yp[:,i],mp[:,i],self.eps)
            ea=outputs['eeg_aux_logits'][l]; ma=outputs['em_aux_logits'][l]
            laux=laux+_masked_bce(ea,y[:,i],mr[:,i],self.eps)+_masked_bce(ma,y[:,i],mr[:,i],self.eps)
            es=ea.detach() if self.det else ea; ms=ma.detach() if self.det else ma
            den=es.abs()+ms.abs()+self.eps
            contrib=torch.stack([es.abs()/den,ms.abs()/den],1)
            err=(outputs['fusion_weights'][l]-contrib).abs().sum(1)
            lcg=lcg+(err*mr[:,i]).sum()/(mr[:,i].sum()+self.eps)
        total=self.lambda_main*(lrec+lpred)+self.lambda_aux*laux+self.lambda_cg*lcg
        return {'loss_total':total,'loss_rec':lrec,'loss_pred':lpred,'loss_aux':laux,'loss_cg':lcg}
