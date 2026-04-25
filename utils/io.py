from pathlib import Path
import json, yaml, numpy as np

def ensure_dir(path):
    p=Path(path); p.mkdir(parents=True, exist_ok=True); return p

def load_yaml(path):
    with open(path,'r',encoding='utf-8') as f: return yaml.safe_load(f)

def save_yaml(obj,path):
    ensure_dir(Path(path).parent)
    with open(path,'w',encoding='utf-8') as f: yaml.safe_dump(obj,f,sort_keys=False)

def save_json(obj,path):
    ensure_dir(Path(path).parent)
    with open(path,'w',encoding='utf-8') as f: json.dump(obj,f,indent=2)

def load_npz(path):
    with np.load(path,allow_pickle=True) as d: return {k:d[k] for k in d.files}

def save_npz(path, **arrays):
    ensure_dir(Path(path).parent); np.savez_compressed(path, **arrays)
