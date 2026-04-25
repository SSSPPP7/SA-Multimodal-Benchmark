import csv, logging
from pathlib import Path

def get_logger(name='MDF_SACN', log_file=None):
    logger=logging.getLogger(name); logger.setLevel(logging.INFO); logger.handlers.clear()
    fmt=logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    sh=logging.StreamHandler(); sh.setFormatter(fmt); logger.addHandler(sh)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh=logging.FileHandler(log_file, encoding='utf-8'); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger

class CSVLogger:
    def __init__(self,path,fieldnames):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True); self.fieldnames=list(fieldnames)
        with self.path.open('w',newline='',encoding='utf-8') as f:
            csv.DictWriter(f,fieldnames=self.fieldnames).writeheader()
    def log(self,row):
        with self.path.open('a',newline='',encoding='utf-8') as f:
            csv.DictWriter(f,fieldnames=self.fieldnames).writerow({k:row.get(k,'') for k in self.fieldnames})
