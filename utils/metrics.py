import numpy as np, torch
LEVELS=("sa1","sa2","sa3")

def binary_metrics_from_logits(logits, labels, mask=None, threshold=0.5):
    logits=np.asarray(logits).reshape(-1); labels=np.asarray(labels).reshape(-1).astype(int)
    mask=np.ones_like(labels,dtype=bool) if mask is None else np.asarray(mask).reshape(-1).astype(bool)
    if mask.sum()==0: return {"acc":np.nan,"recall":np.nan,"precision":np.nan,"f1":np.nan,"n":0}
    probs=1/(1+np.exp(-logits[mask])); pred=(probs>=threshold).astype(int); y=labels[mask]
    tp=int(((pred==1)&(y==1)).sum()); tn=int(((pred==0)&(y==0)).sum())
    fp=int(((pred==1)&(y==0)).sum()); fn=int(((pred==0)&(y==1)).sum())
    acc=(tp+tn)/max(tp+tn+fp+fn,1); rec=tp/max(tp+fn,1); pre=tp/max(tp+fp,1)
    f1=2*pre*rec/max(pre+rec,1e-12)
    return {"acc":acc,"recall":rec,"precision":pre,"f1":f1,"n":int(mask.sum())}

def collect_six_task_metrics(outputs, labels, threshold=0.5):
    y_rec=labels["y_rec"]; y_pred=labels["y_pred"]; mask_pred=labels["mask_pred"]
    mask_rec=labels.get("mask_rec",np.ones_like(y_rec))
    out={}
    for i,lvl in enumerate(LEVELS):
        out[f"{lvl.upper()}-Rec"]=binary_metrics_from_logits(outputs["rec_logits"][lvl], y_rec[:,i], mask_rec[:,i], threshold)
        out[f"{lvl.upper()}-Pred"]=binary_metrics_from_logits(outputs["pred_logits"][lvl], y_pred[:,i], mask_pred[:,i], threshold)
    return out
