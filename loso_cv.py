import argparse
from pathlib import Path
import pandas as pd
from datasets.sa_dataset import SADataset
from train import train_one_fold
from utils.io import load_yaml, ensure_dir, save_json

def flat(r):
    row={'test_subject':r['test_subject'],'best_epoch':r['best_epoch'],'best_val':r['best_val']}
    for t,ms in r['metrics'].items():
        for k,v in ms.items(): row[f'{t}_{k}']=v
    return row

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/default.yaml'); ap.add_argument('--subjects',nargs='*'); ap.add_argument('--out-dir')
    a=ap.parse_args(); cfg=load_yaml(a.config); ds=SADataset(cfg['data']['processed_npz']); subs=a.subjects or list(ds.get_subjects()); out=ensure_dir(a.out_dir or Path(cfg['output_dir'])/'loso')
    rows=[]
    for s in subs:
        r=train_one_fold(cfg,str(s),str(out/f'fold_{s}')); rows.append(flat(r)); pd.DataFrame(rows).to_csv(out/'loso_partial.csv',index=False)
    df=pd.DataFrame(rows); df.to_csv(out/'loso_all_folds.csv',index=False)
    summ={c:{'mean':float(df[c].mean()),'std':float(df[c].std(ddof=1))} for c in df.columns if c.endswith(('_acc','_recall','_precision','_f1'))}
    save_json(summ,str(out/'loso_summary.json')); print(summ)
if __name__=='__main__': main()
