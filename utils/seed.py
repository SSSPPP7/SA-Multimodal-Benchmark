import os, random
import numpy as np
import torch

def set_seed(seed:int=42, deterministic:bool=True)->None:
    random.seed(seed); np.random.seed(seed); os.environ["PYTHONHASHSEED"]=str(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False

def seed_worker(worker_id:int)->None:
    worker_seed=torch.initial_seed()%2**32
    np.random.seed(worker_seed); random.seed(worker_seed)
