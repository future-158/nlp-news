import pickle
import os
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from omegaconf import OmegaConf
import joblib
import tqdm
from statistics import mode
from sklearn.metrics import accuracy_score


cfg = OmegaConf.load('conf/config.yaml')

# index, title,cleanBody,category,result 

# find best model

items = []
for metric_path in cfg.catalog.metric.values():
    item = OmegaConf.to_container(
        OmegaConf.load(metric_path)
    )
    item['model'] = Path(metric_path).stem
    items.append(item)

table = pd.DataFrame(items)


# find best submit file

df = pd.read_csv(cfg.catalog.output.en_base)

submit = pd.read_csv(cfg.catalog.raw.test)

assert (df[df.split =='test'].category.values == submit.category.values).all()

submit['result'] = df[df.split=='test'].result



df








